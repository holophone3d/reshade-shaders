/*
 * 2022 Jake Downs
 */

/*
*
* Based on generic_depth
* Copyright (C) 2021 Patrick Mours
* SPDX-License-Identifier: BSD-3-Clause
*/

#include <imgui.h>
#include <reshade.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <shared_mutex>
#include <unordered_map>

using namespace reshade::api;

static std::shared_mutex s_mutex;

static bool s_disable_intz = false;
// Enable or disable the creation of backup copies at clear operations on the selected depth-stencil
static unsigned int s_preserve_depth_buffers = 0;
// Enable or disable the aspect ratio check from 'check_aspect_ratio' in the detection heuristic
static unsigned int s_use_aspect_ratio_heuristics = 0;

enum class clear_op
{
	clear_depth_stencil_view,
	fullscreen_draw,
	unbind_depth_stencil_view,
};

struct draw_stats
{
	uint32_t vertices = 0;
	uint32_t drawcalls = 0;
	uint32_t drawcalls_indirect = 0;
	viewport last_viewport = {};
};
struct clear_stats : public draw_stats
{
	clear_op clear_op = clear_op::clear_depth_stencil_view;
	bool copied_during_frame = false;
};

struct depth_stencil_info
{
	draw_stats total_stats;
	draw_stats current_stats; // Stats since last clear operation
	std::vector<clear_stats> clears;
	bool copied_during_frame = false;
};

struct depth_stencil_hash
{
	inline size_t operator()(resource value) const
	{
		// Simply use the handle (which is usually a pointer) as hash value (with some bits shaved off due to pointer alignment)
		return static_cast<size_t>(value.handle >> 4);
	}
};

struct __declspec(uuid("ad059cc1-c3ad-4cef-a4a9-401f672c6c37")) state_tracking
{
	viewport current_viewport = {};
	resource current_depth_stencil = { 0 };
	std::unordered_map<resource, depth_stencil_info, depth_stencil_hash> counters_per_used_depth_stencil;
	bool first_draw_since_bind = true;
	draw_stats best_copy_stats;

	state_tracking()
	{
		// Reserve some space upfront to avoid rehashing during command recording
		counters_per_used_depth_stencil.reserve(32);
	}

	void reset()
	{
		reset_on_present();
		current_depth_stencil = { 0 };
	}
	void reset_on_present()
	{
		best_copy_stats = { 0, 0 };
		counters_per_used_depth_stencil.clear();
	}

	void merge(const state_tracking &source)
	{
		// Executing a command list in a different command list inherits state
		current_depth_stencil = source.current_depth_stencil;

		if (source.best_copy_stats.vertices >= best_copy_stats.vertices)
			best_copy_stats = source.best_copy_stats;

		if (source.counters_per_used_depth_stencil.empty())
			return;

		counters_per_used_depth_stencil.reserve(source.counters_per_used_depth_stencil.size());
		for (const auto &[depth_stencil_handle, snapshot] : source.counters_per_used_depth_stencil)
		{
			depth_stencil_info &target_snapshot = counters_per_used_depth_stencil[depth_stencil_handle];
			target_snapshot.total_stats.vertices += snapshot.total_stats.vertices;
			target_snapshot.total_stats.drawcalls += snapshot.total_stats.drawcalls;
			target_snapshot.total_stats.drawcalls_indirect += snapshot.total_stats.drawcalls_indirect;
			target_snapshot.current_stats.vertices += snapshot.current_stats.vertices;
			target_snapshot.current_stats.drawcalls += snapshot.current_stats.drawcalls;
			target_snapshot.current_stats.drawcalls_indirect += snapshot.current_stats.drawcalls_indirect;

			target_snapshot.clears.insert(target_snapshot.clears.end(), snapshot.clears.begin(), snapshot.clears.end());

			target_snapshot.copied_during_frame |= snapshot.copied_during_frame;
		}
	}
};

struct __declspec(uuid("7c6363c7-f94e-437a-9160-141782c44a98")) generic_depth_data
{
	// The depth-stencil resource that is currently selected as being the main depth target
	resource selected_depth_stencil = { 0 };

	// Resource used to override automatic depth-stencil selection
	resource override_depth_stencil = { 0 };

	// The current depth shader resource view bound to shaders
	// This can be created from either the selected depth-stencil resource (if it supports shader access) or from a backup resource
	resource_view selected_shader_resource = { 0 };

	// True when the shader resource view was created from the backup resource, false when it was created from the original depth-stencil
	bool using_backup_texture = false;

	std::unordered_map<resource, unsigned int, depth_stencil_hash> display_count_per_depth_stencil;
};

struct depth_stencil_backup
{
	// The number of effect runtimes referencing this backup
	size_t references = 1;

	// A resource used as target for a backup copy of this depth-stencil
	resource backup_texture = { 0 };

	// The depth-stencil that should be copied from
	resource depth_stencil_resource = { 0 };

	// Set to zero for automatic detection, otherwise will use the clear operation at the specific index within a frame
	size_t force_clear_index = 0;

	// Frame dimensions of the last effect runtime this backup was used with
	uint32_t frame_width = 0;
	uint32_t frame_height = 0;
};

struct __declspec(uuid("e006e162-33ac-4b9f-b10f-0e15335c7bdb")) generic_depth_device_data
{
	// List of queues created for this device
	std::vector<command_queue *> queues;

	// List of resources that were deleted this frame
	std::vector<resource> destroyed_resources;

	// List of resources that are enqueued for delayed destruction in the future
	std::vector<std::pair<resource, int>> delayed_destroy_resources;

	// List of all encountered depth-stencils of the last frame
	std::vector<std::pair<resource, depth_stencil_info>> current_depth_stencil_list;

	// List of depth-stencils that should be tracked throughout each frame and potentially be backed up during clear operations
	std::vector<depth_stencil_backup> depth_stencil_backups;

	depth_stencil_backup *find_depth_stencil_backup(resource resource)
	{
		for (depth_stencil_backup &backup : depth_stencil_backups)
			if (backup.depth_stencil_resource == resource)
				return &backup;
		return nullptr;
	}

	depth_stencil_backup *track_depth_stencil_for_backup(device *device, resource resource, resource_desc desc)
	{
		const auto it = std::find_if(depth_stencil_backups.begin(), depth_stencil_backups.end(),
			[resource](const depth_stencil_backup &existing) { return existing.depth_stencil_resource == resource; });
		if (it != depth_stencil_backups.end())
		{
			it->references++;
			return &(*it);
		}

		depth_stencil_backup &backup = depth_stencil_backups.emplace_back();
		backup.depth_stencil_resource = resource;

		desc.type = resource_type::texture_2d;
		desc.heap = memory_heap::gpu_only;
		desc.usage = resource_usage::shader_resource | resource_usage::copy_dest;

		if (device->get_api() == device_api::d3d9)
			desc.texture.format = format::r32_float; // D3DFMT_R32F, since INTZ does not support D3DUSAGE_RENDERTARGET which is required for copying
		// Use depth format as-is in OpenGL and Vulkan, since those are valid for shader resource views there
		else if (device->get_api() != device_api::opengl && device->get_api() != device_api::vulkan)
			desc.texture.format = format_to_typeless(desc.texture.format);

		// First try to revive a backup resource that was previously enqueued for delayed destruction
		for (auto delayed_destroy_it = delayed_destroy_resources.begin(); delayed_destroy_it != delayed_destroy_resources.end(); ++delayed_destroy_it)
		{
			const resource_desc delayed_destroy_desc = device->get_resource_desc(delayed_destroy_it->first);

			if (desc.texture.width == delayed_destroy_desc.texture.width && desc.texture.height == delayed_destroy_desc.texture.height && desc.texture.format == delayed_destroy_desc.texture.format)
			{
				backup.backup_texture = delayed_destroy_it->first;
				delayed_destroy_resources.erase(delayed_destroy_it);
				return &backup;
			}
		}

		if (device->create_resource(desc, nullptr, resource_usage::copy_dest, &backup.backup_texture))
			device->set_resource_name(backup.backup_texture, "ReShade depth backup texture");
		else
			reshade::log_message(1, "Failed to create backup depth-stencil texture!");

		return &backup;
	}

	void untrack_depth_stencil(resource resource)
	{
		const auto it = std::find_if(depth_stencil_backups.begin(), depth_stencil_backups.end(),
			[resource](const depth_stencil_backup &existing) { return existing.depth_stencil_resource == resource; });
		if (it == depth_stencil_backups.end() || --it->references != 0)
			return;

		depth_stencil_backup &backup = *it;

		if (backup.backup_texture != 0)
		{
			// Do not destroy backup texture immediately since it may still be referenced by a command list that is in flight or was prerecorded
			// Instead enqueue it for delayed destruction in the future
			delayed_destroy_resources.emplace_back(backup.backup_texture, 50); // Destroy after 50 frames
		}

		depth_stencil_backups.erase(it);
	}
};

// Checks whether the aspect ratio of the two sets of dimensions is similar or not
static bool check_aspect_ratio(float width_to_check, float height_to_check, uint32_t width, uint32_t height)
{
	if (width_to_check == 0.0f || height_to_check == 0.0f)
		return true;

	const float w = static_cast<float>(width);
	float w_ratio = w / width_to_check;
	const float h = static_cast<float>(height);
	float h_ratio = h / height_to_check;
	const float aspect_ratio = (w / h) - (static_cast<float>(width_to_check) / height_to_check);

	// Accept if dimensions are similar in value or almost exact multiples
	return std::fabs(aspect_ratio) <= 0.1f && ((w_ratio <= 1.85f && w_ratio >= 0.5f && h_ratio <= 1.85f && h_ratio >= 0.5f) || (s_use_aspect_ratio_heuristics == 2 && std::modf(w_ratio, &w_ratio) <= 0.02f && std::modf(h_ratio, &h_ratio) <= 0.02f));
}

static void on_clear_depth_impl(command_list *cmd_list, state_tracking &state, resource depth_stencil, clear_op op)
{
	if (depth_stencil == 0)
		return;

	device *const device = cmd_list->get_device();

	depth_stencil_backup *const depth_stencil_backup = device->get_private_data<generic_depth_device_data>().find_depth_stencil_backup(depth_stencil);
	if (depth_stencil_backup == nullptr || depth_stencil_backup->backup_texture == 0)
		return;

	bool do_copy = true;
	depth_stencil_info &counters = state.counters_per_used_depth_stencil[depth_stencil];

	// Ignore clears when there was no meaningful workload (e.g. at the start of a frame)
	if (counters.current_stats.drawcalls == 0)
		return;

	// Ignore clears when the last viewport rendered to only affected a small subset of the depth-stencil (fixes flickering in some games)
	switch (op)
	{
	case clear_op::clear_depth_stencil_view:
		// Mirror's Edge and Portal occasionally render something into a small viewport (16x16 in Mirror's Edge, 512x512 in Portal to render underwater geometry)
		do_copy = counters.current_stats.last_viewport.width > 1024 || (counters.current_stats.last_viewport.width == 0 || depth_stencil_backup->frame_width <= 1024);
		break;
	case clear_op::fullscreen_draw:
		// Mass Effect 3 in Mass Effect Legendary Edition sometimes uses a larger common depth buffer for shadow map and scene rendering, where the former uses a 1024x1024 viewport and the latter uses a viewport matching the render resolution
		do_copy = check_aspect_ratio(counters.current_stats.last_viewport.width, counters.current_stats.last_viewport.height, depth_stencil_backup->frame_width, depth_stencil_backup->frame_height);
		break;
	case clear_op::unbind_depth_stencil_view:
		break;
	}

	if (do_copy)
	{
		if (op != clear_op::unbind_depth_stencil_view)
		{
			// If clear index override is set to zero, always copy any suitable buffers
			if (depth_stencil_backup->force_clear_index == 0)
			{
				// Use greater equals operator here to handle case where the same scene is first rendered into a shadow map and then for real (e.g. Mirror's Edge main menu)
				do_copy = counters.current_stats.vertices >= state.best_copy_stats.vertices || (op == clear_op::fullscreen_draw && counters.current_stats.drawcalls >= state.best_copy_stats.drawcalls);
			}
			else if (std::numeric_limits<size_t>::max() == depth_stencil_backup->force_clear_index)
			{
				// Special case for Garry's Mod which chooses the last clear operation that has a high workload
				do_copy = counters.current_stats.vertices >= 5000;
			}
			else
			{
				// This is not really correct, since clears may accumulate over multiple command lists, but it's unlikely that the same depth-stencil is used in more than one
				do_copy = counters.clears.size() == (depth_stencil_backup->force_clear_index - 1);
			}

			counters.clears.push_back({ counters.current_stats, op, do_copy });
		}

		// Make a backup copy of the depth texture before it is cleared
		if (do_copy)
		{
			state.best_copy_stats = counters.current_stats;

			// A resource has to be in this state for a clear operation, so can assume it here
			cmd_list->barrier(depth_stencil, resource_usage::depth_stencil_write, resource_usage::copy_source);
			cmd_list->copy_resource(depth_stencil, depth_stencil_backup->backup_texture);
			cmd_list->barrier(depth_stencil, resource_usage::copy_source, resource_usage::depth_stencil_write);

			counters.copied_during_frame = true;
		}
	}

	// Reset draw call stats for clears
	counters.current_stats = { 0, 0 };
}

static void update_effect_runtime(effect_runtime *runtime)
{
	const generic_depth_data &instance = runtime->get_private_data<generic_depth_data>();

	runtime->update_texture_bindings("ORIG_DEPTH", instance.selected_shader_resource);

	runtime->enumerate_uniform_variables(nullptr, [&instance](effect_runtime *runtime, auto variable) {
		char source[32] = "";
		if (runtime->get_annotation_string_from_uniform_variable(variable, "source", source) && std::strcmp(source, "bufready_depth") == 0)
			runtime->set_uniform_value_bool(variable, instance.selected_shader_resource != 0);
	});

	resource_view srv, srv_srgb;

	effect_texture_variable ModifiedDepthTex_handle = runtime->find_texture_variable("Citra.fx", "ModifiedDepthTex");
	runtime->get_texture_binding(ModifiedDepthTex_handle, &srv, &srv_srgb);

	runtime->update_texture_bindings("DEPTH", srv, srv_srgb);
}

static void on_init_device(device *device)
{
	device->create_private_data<generic_depth_device_data>();

	reshade::config_get_value(nullptr, "DEPTH", "DisableINTZ", s_disable_intz);
	reshade::config_get_value(nullptr, "DEPTH", "DepthCopyBeforeClears", s_preserve_depth_buffers);
	reshade::config_get_value(nullptr, "DEPTH", "UseAspectRatioHeuristics", s_use_aspect_ratio_heuristics);
}
static void on_init_command_list(command_list *cmd_list)
{
	cmd_list->create_private_data<state_tracking>();
}
static void on_init_command_queue(command_queue *cmd_queue)
{
	cmd_queue->create_private_data<state_tracking>();

	if ((cmd_queue->get_type() & command_queue_type::graphics) == 0)
		return;

	auto &device_data = cmd_queue->get_device()->get_private_data<generic_depth_device_data>();
	device_data.queues.push_back(cmd_queue);
}
static void on_init_effect_runtime(effect_runtime *runtime)
{
	runtime->create_private_data<generic_depth_data>();
}
static void on_destroy_device(device *device)
{
	auto &device_data = device->get_private_data<generic_depth_device_data>();

	// Destroy any remaining resources
	for (const auto &[resource, _] : device_data.delayed_destroy_resources)
	{
		device->destroy_resource(resource);
	}

	for (depth_stencil_backup &depth_stencil_backup : device_data.depth_stencil_backups)
	{
		if (depth_stencil_backup.backup_texture != 0)
			device->destroy_resource(depth_stencil_backup.backup_texture);
	}

	device->destroy_private_data<generic_depth_device_data>();
}
static void on_destroy_command_list(command_list *cmd_list)
{
	cmd_list->destroy_private_data<state_tracking>();
}
static void on_destroy_command_queue(command_queue *cmd_queue)
{
	cmd_queue->destroy_private_data<state_tracking>();

	auto &device_data = cmd_queue->get_device()->get_private_data<generic_depth_device_data>();
	device_data.queues.erase(std::remove(device_data.queues.begin(), device_data.queues.end(), cmd_queue), device_data.queues.end());
}
static void on_destroy_effect_runtime(effect_runtime *runtime)
{
	device *const device = runtime->get_device();
	generic_depth_data &data = runtime->get_private_data<generic_depth_data>();

	if (data.selected_shader_resource != 0)
		device->destroy_resource_view(data.selected_shader_resource);

	runtime->destroy_private_data<generic_depth_data>();
}

static bool on_create_resource(device *device, resource_desc &desc, subresource_data *, resource_usage)
{
	if (desc.type != resource_type::surface && desc.type != resource_type::texture_2d)
		return false; // Skip resources that are not 2D textures
	if (desc.texture.samples != 1 || (desc.usage & resource_usage::depth_stencil) == 0 || desc.texture.format == format::s8_uint)
		return false; // Skip MSAA textures and resources that are not used as depth buffers

	switch (device->get_api())
	{
	case device_api::d3d9:
		if (s_disable_intz)
			return false;
		// Skip textures that are sampled as PCF shadow maps (see https://aras-p.info/texts/D3D9GPUHacks.html#shadowmap) using hardware support, since changing format would break that
		if (desc.type == resource_type::texture_2d && (desc.texture.format == format::d16_unorm || desc.texture.format == format::d24_unorm_x8_uint || desc.texture.format == format::d24_unorm_s8_uint))
			return false;
		// Skip small textures that are likely just shadow maps too (fixes a hang in Dragon's Dogma: Dark Arisen when changing areas)
		if (desc.texture.width <= 512)
			return false;
		// Replace texture format with special format that supports normal sampling (see https://aras-p.info/texts/D3D9GPUHacks.html#depth)
		desc.texture.format = format::intz;
		desc.usage |= resource_usage::shader_resource;
		break;
	case device_api::d3d10:
	case device_api::d3d11:
		// Allow shader access to images that are used as depth-stencil attachments
		desc.texture.format = format_to_typeless(desc.texture.format);
		desc.usage |= resource_usage::shader_resource;
		break;
	case device_api::d3d12:
	case device_api::vulkan:
		// D3D12 and Vulkan always use backup texture, but need to be able to copy to it
		desc.usage |= resource_usage::copy_source;
		break;
	case device_api::opengl:
		// No need to change anything in OpenGL
		return false;
	}

	return true;
}
static bool on_create_resource_view(device *device, resource resource, resource_usage usage_type, resource_view_desc &desc)
{
	// A view cannot be created with a typeless format (which was set in 'on_create_resource' above), so fix it in case defaults are used
	if ((device->get_api() != device_api::d3d10 && device->get_api() != device_api::d3d11) || desc.format != format::unknown)
		return false;

	const resource_desc texture_desc = device->get_resource_desc(resource);
	// Only non-MSAA textures where modified, so skip all others
	if (texture_desc.texture.samples != 1 || (texture_desc.usage & resource_usage::depth_stencil) == 0)
		return false;

	switch (usage_type)
	{
	case resource_usage::depth_stencil:
		desc.format = format_to_depth_stencil_typed(texture_desc.texture.format);
		break;
	case resource_usage::shader_resource:
		desc.format = format_to_default_typed(texture_desc.texture.format);
		break;
	}

	// Only need to set the rest of the fields if the application did not pass in a valid description already
	if (desc.type == resource_view_type::unknown)
	{
		desc.type = texture_desc.texture.depth_or_layers > 1 ? resource_view_type::texture_2d_array : resource_view_type::texture_2d;
		desc.texture.first_level = 0;
		desc.texture.level_count = (usage_type == resource_usage::shader_resource) ? UINT32_MAX : 1;
		desc.texture.first_layer = 0;
		desc.texture.layer_count = (usage_type == resource_usage::shader_resource) ? UINT32_MAX : 1;
	}

	return true;
}
static void on_destroy_resource(device *device, resource resource)
{
	auto &device_data = device->get_private_data<generic_depth_device_data>();

	// In some cases the 'destroy_device' event may be called before all resources have been destroyed
	// The state tracking context would have been destroyed already in that case, so return early if it does not exist
	if (std::addressof(device_data) == nullptr)
		return;

	std::unique_lock<std::shared_mutex> lock(s_mutex);

	device_data.destroyed_resources.push_back(resource);

	// Remove this destroyed resource from the list of tracked depth-stencil resources
	const auto it = std::find_if(device_data.current_depth_stencil_list.begin(), device_data.current_depth_stencil_list.end(),
		[resource](const auto &current) { return current.first == resource; });
	if (it != device_data.current_depth_stencil_list.end())
	{
		const bool copied_during_frame = it->second.copied_during_frame;

		device_data.current_depth_stencil_list.erase(it);

		lock.unlock();

		// This is bad ... the resource may still be in use by an effect on the GPU and destroying it would crash it
		// Try to mitigate that somehow by delaying this thread a little to hopefully give the GPU enough time to catch up before the resource memory is deallocated
		if (device->get_api() == device_api::d3d12 || device->get_api() == device_api::vulkan)
		{
			reshade::log_message(2, "A depth-stencil resource was destroyed while still being tracked.");

			if (!copied_during_frame)
				Sleep(250);
		}
	}
}

static bool on_draw(command_list *cmd_list, uint32_t vertices, uint32_t instances, uint32_t, uint32_t)
{
	auto &state = cmd_list->get_private_data<state_tracking>();
	if (state.current_depth_stencil == 0)
		return false; // This is a draw call with no depth-stencil bound

	// Check if this draw call likely represets a fullscreen rectangle (two triangles), which would clear the depth-stencil
	const bool fullscreen_draw = vertices == 6 && instances == 1;
	if (fullscreen_draw &&
		s_preserve_depth_buffers == 2 &&
		state.first_draw_since_bind &&
		// But ignore that in Vulkan (since it is invalid to copy a resource inside an active render pass)
		cmd_list->get_device()->get_api() != device_api::vulkan)
		on_clear_depth_impl(cmd_list, state, state.current_depth_stencil, clear_op::fullscreen_draw);

	state.first_draw_since_bind = false;

	depth_stencil_info &counters = state.counters_per_used_depth_stencil[state.current_depth_stencil];
	counters.total_stats.vertices += vertices * instances;
	counters.total_stats.drawcalls += 1;
	counters.current_stats.vertices += vertices * instances;
	counters.current_stats.drawcalls += 1;

	// Skip updating last viewport for fullscreen draw calls, to prevent a clear operation in Prince of Persia: The Sands of Time from getting filtered out
	if (!fullscreen_draw)
		counters.current_stats.last_viewport = state.current_viewport;

	return false;
}
static bool on_draw_indexed(command_list *cmd_list, uint32_t indices, uint32_t instances, uint32_t, int32_t, uint32_t)
{
	on_draw(cmd_list, indices, instances, 0, 0);

	return false;
}
static bool on_draw_indirect(command_list *cmd_list, indirect_command type, resource, uint64_t, uint32_t draw_count, uint32_t)
{
	if (type == indirect_command::dispatch)
		return false;

	auto &state = cmd_list->get_private_data<state_tracking>();
	if (state.current_depth_stencil == 0)
		return false; // This is a draw call with no depth-stencil bound

	depth_stencil_info &counters = state.counters_per_used_depth_stencil[state.current_depth_stencil];
	counters.total_stats.drawcalls += draw_count;
	counters.total_stats.drawcalls_indirect += draw_count;
	counters.current_stats.drawcalls += draw_count;
	counters.current_stats.drawcalls_indirect += draw_count;
	counters.current_stats.last_viewport = state.current_viewport;

	return false;
}

static void on_bind_viewport(command_list *cmd_list, uint32_t first, uint32_t count, const viewport *viewport)
{
	if (first != 0 || count == 0)
		return; // Only interested in the main viewport

	auto &state = cmd_list->get_private_data<state_tracking>();
	state.current_viewport = viewport[0];
}
static void on_bind_depth_stencil(command_list *cmd_list, uint32_t, const resource_view *, resource_view depth_stencil_view)
{
	auto &state = cmd_list->get_private_data<state_tracking>();

	const resource depth_stencil = (depth_stencil_view != 0) ? cmd_list->get_device()->get_resource_from_view(depth_stencil_view) : resource{ 0 };

	if (depth_stencil != state.current_depth_stencil)
	{
		if (depth_stencil != 0)
			state.first_draw_since_bind = true;

		// Make a backup of the depth texture before it is used differently, since in D3D12 or Vulkan the underlying memory may be aliased to a different resource, so cannot just access it at the end of the frame
		if (s_preserve_depth_buffers == 2 &&
			state.current_depth_stencil != 0 && depth_stencil == 0 && (
			cmd_list->get_device()->get_api() == device_api::d3d12 || cmd_list->get_device()->get_api() == device_api::vulkan))
			on_clear_depth_impl(cmd_list, state, state.current_depth_stencil, clear_op::unbind_depth_stencil_view);
	}

	state.current_depth_stencil = depth_stencil;
}
static bool on_clear_depth_stencil(command_list *cmd_list, resource_view dsv, const float *depth, const uint8_t *, uint32_t, const rect *)
{
	// Ignore clears that do not affect the depth buffer (stencil clears)
	if (depth != nullptr && s_preserve_depth_buffers)
	{
		auto &state = cmd_list->get_private_data<state_tracking>();

		const resource depth_stencil = cmd_list->get_device()->get_resource_from_view(dsv);

		// Note: This does not work when called from 'vkCmdClearAttachments', since it is invalid to copy a resource inside an active render pass
		on_clear_depth_impl(cmd_list, state, depth_stencil, clear_op::clear_depth_stencil_view);
	}

	return false;
}
static void on_begin_render_pass_with_depth_stencil(command_list *cmd_list, uint32_t, const render_pass_render_target_desc *, const render_pass_depth_stencil_desc *depth_stencil_desc)
{
	if (depth_stencil_desc != nullptr && depth_stencil_desc->depth_load_op == render_pass_load_op::clear)
	{
		on_clear_depth_stencil(cmd_list, depth_stencil_desc->view, &depth_stencil_desc->clear_depth, nullptr, 0, nullptr);

		// Prevent 'on_bind_depth_stencil' from copying depth buffer again
		auto &state = cmd_list->get_private_data<state_tracking>();
		state.current_depth_stencil = { 0 };
	}

	// If render pass has depth store operation set to 'discard', any copy performed after the render pass will likely contain broken data, so can only hope that the depth buffer can be copied before that ...

	on_bind_depth_stencil(cmd_list, 0, nullptr, depth_stencil_desc != nullptr ? depth_stencil_desc->view : resource_view{});
}

static void on_reset(command_list *cmd_list)
{
	auto &target_state = cmd_list->get_private_data<state_tracking>();
	target_state.reset();
}
static void on_execute_primary(command_queue *queue, command_list *cmd_list)
{
	auto &target_state = queue->get_private_data<state_tracking>();
	const auto &source_state = cmd_list->get_private_data<state_tracking>();

	// Skip merging state when this execution event is just the immediate command list getting flushed
	if (std::addressof(target_state) != std::addressof(source_state))
	{
		target_state.merge(source_state);
	}
}
static void on_execute_secondary(command_list *cmd_list, command_list *secondary_cmd_list)
{
	auto &target_state = cmd_list->get_private_data<state_tracking>();
	const auto &source_state = secondary_cmd_list->get_private_data<state_tracking>();

	// If this is a secondary command list that was recorded without a depth-stencil binding, but is now executed using a depth-stencil binding, handle it as if an indirect draw call was performed to ensure the depth-stencil is tracked
	if (target_state.current_depth_stencil != 0 && source_state.current_depth_stencil == 0 && source_state.counters_per_used_depth_stencil.empty())
	{
		target_state.current_viewport = source_state.current_viewport;

		on_draw_indirect(cmd_list, indirect_command::draw, { 0 }, 0, 1, 0);
	}
	else
	{
		target_state.merge(source_state);
	}
}

static void on_present(command_queue *, swapchain *swapchain, const rect *, const rect *, uint32_t, const rect *)
{
	device *const device = swapchain->get_device();
	generic_depth_device_data &device_data = device->get_private_data<generic_depth_device_data>();

	const std::unique_lock<std::shared_mutex> lock(s_mutex);

	// Merge state from all graphics queues
	state_tracking queue_state;
	for (command_queue *const queue : device_data.queues)
		queue_state.merge(queue->get_private_data<state_tracking>());

	// Only update device list if there are any depth-stencils, otherwise this may be a second present call (at which point 'reset_on_present' already cleared out the queue list in the first present call)
	if (queue_state.counters_per_used_depth_stencil.empty())
		return;

	// Also skip update when there has been very little activity (special case for emulators like PCSX2 which may present more often than they render a frame)
	if (queue_state.counters_per_used_depth_stencil.size() == 1 && queue_state.counters_per_used_depth_stencil.begin()->second.total_stats.drawcalls <= 8)
		return;

	device_data.current_depth_stencil_list.clear();
	device_data.current_depth_stencil_list.reserve(queue_state.counters_per_used_depth_stencil.size());

	for (const auto &[resource, snapshot] : queue_state.counters_per_used_depth_stencil)
	{
		if (snapshot.total_stats.drawcalls == 0)
			continue; // Skip unused

		if (std::find(device_data.destroyed_resources.begin(), device_data.destroyed_resources.end(), resource) != device_data.destroyed_resources.end())
			continue; // Skip resources that were destroyed by the application

		// Save to current list of depth-stencils on the device, so that it can be displayed in the GUI
		device_data.current_depth_stencil_list.emplace_back(resource, snapshot);
	}

	for (command_queue *const queue : device_data.queues)
		queue->get_private_data<state_tracking>().reset_on_present();

	device_data.destroyed_resources.clear();

	// Destroy resources that were enqueued for delayed destruction and have reached the targeted number of passed frames
	for (auto it = device_data.delayed_destroy_resources.begin(); it != device_data.delayed_destroy_resources.end();)
	{
		if (--it->second == 0)
		{
			device->destroy_resource(it->first);

			it = device_data.delayed_destroy_resources.erase(it);
		}
		else
		{
			++it;
		}
	}
}

static void on_begin_render_effects(effect_runtime *runtime, command_list *cmd_list, resource_view, resource_view)
{
	device *const device = runtime->get_device();
	generic_depth_data &data = runtime->get_private_data<generic_depth_data>();
	generic_depth_device_data &device_data = device->get_private_data<generic_depth_device_data>();

	resource best_match = { 0 };
	resource_desc best_match_desc;
	const depth_stencil_info *best_snapshot = nullptr;

	uint32_t frame_width, frame_height;
	runtime->get_screenshot_width_and_height(&frame_width, &frame_height);

	std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto current_depth_stencil_list = device_data.current_depth_stencil_list;
	// Unlock while calling into device below, since device may hold a lock itself and that then can deadlock another thread that calls into 'on_destroy_resource' from the device holding that lock
	lock.unlock();

	for (auto &[resource, snapshot] : current_depth_stencil_list)
	{
		const resource_desc desc = device->get_resource_desc(resource);
		if (desc.texture.samples > 1)
			continue; // Ignore MSAA textures, since they would need to be resolved first

		if (s_use_aspect_ratio_heuristics && !check_aspect_ratio(static_cast<float>(desc.texture.width), static_cast<float>(desc.texture.height), frame_width, frame_height))
			continue; // Not a good fit

		if (best_snapshot == nullptr || (snapshot.total_stats.drawcalls_indirect < (snapshot.total_stats.drawcalls / 3) ?
			// Choose snapshot with the most vertices, since that is likely to contain the main scene
			snapshot.total_stats.vertices > best_snapshot->total_stats.vertices :
		// Or check draw calls, since vertices may not be accurate if application is using indirect draw calls
		snapshot.total_stats.drawcalls > best_snapshot->total_stats.drawcalls))
		{
			best_match = resource;
			best_match_desc = desc;
			best_snapshot = &snapshot;
		}
	}

	if (data.override_depth_stencil != 0)
	{
		const auto it = std::find_if(current_depth_stencil_list.begin(), current_depth_stencil_list.end(),
			[resource = data.override_depth_stencil](const auto &current) { return current.first == resource; });
		if (it != current_depth_stencil_list.end())
		{
			best_match = it->first;
			best_match_desc = device->get_resource_desc(it->first);
			best_snapshot = &it->second;
		}
	}

	if (best_match != 0)
	{
		const device_api api = device->get_api();

		depth_stencil_backup *depth_stencil_backup = device_data.find_depth_stencil_backup(best_match);

		if (best_match != data.selected_depth_stencil || data.selected_shader_resource == 0 || (s_preserve_depth_buffers && depth_stencil_backup == nullptr))
		{
			// Destroy previous resource view, since the underlying resource has changed
			if (data.selected_shader_resource != 0)
			{
				runtime->get_command_queue()->wait_idle(); // Ensure resource view is no longer in-use before destroying it
				device->destroy_resource_view(data.selected_shader_resource);

				device_data.untrack_depth_stencil(data.selected_depth_stencil);
			}

			data.using_backup_texture = false;
			data.selected_depth_stencil = best_match;
			data.selected_shader_resource = { 0 };

			// Create two-dimensional resource view to the first level and layer of the depth-stencil resource
			resource_view_desc srv_desc(api != device_api::opengl && api != device_api::vulkan ? format_to_default_typed(best_match_desc.texture.format) : best_match_desc.texture.format);

			// Need to create backup texture only if doing backup copies or original resource does not support shader access (which is necessary for binding it to effects)
			// Also always create a backup texture in D3D12 or Vulkan to circument problems in case application makes use of resource aliasing
			if (s_preserve_depth_buffers || (best_match_desc.usage & resource_usage::shader_resource) == 0 || (api == device_api::d3d12 || api == device_api::vulkan))
			{
				depth_stencil_backup = device_data.track_depth_stencil_for_backup(device, best_match, best_match_desc);

				// Abort in case backup texture creation failed
				if (depth_stencil_backup->backup_texture == 0)
					return;

				depth_stencil_backup->frame_width = frame_width;
				depth_stencil_backup->frame_height = frame_height;

				if (s_preserve_depth_buffers)
					reshade::config_get_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", depth_stencil_backup->force_clear_index);
				else
					depth_stencil_backup->force_clear_index = 0;

				if (api == device_api::d3d9)
					srv_desc.format = format::r32_float; // Same format as backup texture, as set in 'track_depth_stencil_for_backup'

				if (!device->create_resource_view(depth_stencil_backup->backup_texture, resource_usage::shader_resource, srv_desc, &data.selected_shader_resource))
					return;

				data.using_backup_texture = true;
			}
			else
			{
				if (!device->create_resource_view(best_match, resource_usage::shader_resource, srv_desc, &data.selected_shader_resource))
					return;
			}

			update_effect_runtime(runtime);
		}

		if (data.using_backup_texture)
		{
			assert(depth_stencil_backup != nullptr && depth_stencil_backup->backup_texture != 0 && best_snapshot != nullptr);
			const resource backup_texture = depth_stencil_backup->backup_texture;

			// Copy to backup texture unless already copied during the current frame
			if (!best_snapshot->copied_during_frame && (best_match_desc.usage & resource_usage::copy_source) != 0)
			{
				// Ensure barriers are not created with 'D3D12_RESOURCE_STATE_[...]_SHADER_RESOURCE' when resource has 'D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE' flag set
				const resource_usage old_state = best_match_desc.usage & (resource_usage::depth_stencil | resource_usage::shader_resource);

				lock.lock();
				const auto it = std::find_if(device_data.current_depth_stencil_list.begin(), device_data.current_depth_stencil_list.end(),
					[best_match](const auto &current) { return current.first == best_match; });
				// Indicate that the copy is now being done, so it is not repeated in case effects are rendered by another runtime (e.g. when there are multiple present calls in a frame)
				if (it != device_data.current_depth_stencil_list.end())
					it->second.copied_during_frame = true;
				else
					// Resource disappeared from the current depth-stencil list between earlier in this function and now, which indicates that it was destroyed in the meantime
					return;
				lock.unlock();

				cmd_list->barrier(best_match, old_state, resource_usage::copy_source);
				cmd_list->copy_resource(best_match, backup_texture);
				cmd_list->barrier(best_match, resource_usage::copy_source, old_state);
			}

			cmd_list->barrier(backup_texture, resource_usage::copy_dest, resource_usage::shader_resource);
		}
		else
		{
			// Unset current depth-stencil view, in case it is bound to an effect as a shader resource (which will fail if it is still bound on output)
			if (api <= device_api::d3d11)
				cmd_list->bind_render_targets_and_depth_stencil(0, nullptr);

			cmd_list->barrier(best_match, resource_usage::depth_stencil | resource_usage::shader_resource, resource_usage::shader_resource);
		}
	}
	else
	{
		// Unset any existing depth-stencil selected in previous frames
		if (data.selected_depth_stencil != 0)
		{
			if (data.selected_shader_resource != 0)
			{
				runtime->get_command_queue()->wait_idle(); // Ensure resource view is no longer in-use before destroying it
				device->destroy_resource_view(data.selected_shader_resource);

				device_data.untrack_depth_stencil(data.selected_depth_stencil);
			}

			data.using_backup_texture = false;
			data.selected_depth_stencil = { 0 };
			data.selected_shader_resource = { 0 };

			update_effect_runtime(runtime);
		}
	}
}
static void on_finish_render_effects(effect_runtime *runtime, command_list *cmd_list, resource_view, resource_view)
{
	const generic_depth_data &data = runtime->get_private_data<generic_depth_data>();

	if (data.selected_shader_resource != 0)
	{
		if (data.using_backup_texture)
		{
			const resource backup_texture = runtime->get_device()->get_resource_from_view(data.selected_shader_resource);
			cmd_list->barrier(backup_texture, resource_usage::shader_resource, resource_usage::copy_dest);
		}
		else
		{
			cmd_list->barrier(data.selected_depth_stencil, resource_usage::shader_resource, resource_usage::depth_stencil | resource_usage::shader_resource);
		}
	}
}

static inline const char *format_to_string(format format) {
	switch (format)
	{
	case format::d16_unorm:
	case format::r16_typeless:
		return "D16  ";
	case format::d16_unorm_s8_uint:
		return "D16S8";
	case format::d24_unorm_x8_uint:
		return "D24X8";
	case format::d24_unorm_s8_uint:
	case format::r24_g8_typeless:
		return "D24S8";
	case format::d32_float:
	case format::r32_float:
	case format::r32_typeless:
		return "D32  ";
	case format::d32_float_s8_uint:
	case format::r32_g8_typeless:
		return "D32S8";
	case format::intz:
		return "INTZ ";
	default:
		return "     ";
	}
}

static void draw_settings_overlay(effect_runtime *runtime)
{
	device *const device = runtime->get_device();
	generic_depth_data &data = runtime->get_private_data<generic_depth_data>();
	generic_depth_device_data &device_data = device->get_private_data<generic_depth_device_data>();

	bool force_reset = false;

	if (bool use_aspect_ratio_heuristics = s_use_aspect_ratio_heuristics != 0;
		ImGui::Checkbox("Use aspect ratio heuristics", &use_aspect_ratio_heuristics))
	{
		s_use_aspect_ratio_heuristics = use_aspect_ratio_heuristics ? 1 : 0;
		reshade::config_set_value(nullptr, "DEPTH", "UseAspectRatioHeuristics", s_use_aspect_ratio_heuristics);
		force_reset = true;
	}

	if (s_use_aspect_ratio_heuristics)
	{
		if (bool use_aspect_ratio_heuristics_ex = s_use_aspect_ratio_heuristics == 2;
			ImGui::Checkbox("Use extended aspect ratio heuristics (for DLSS or resolution scaling)", &use_aspect_ratio_heuristics_ex))
		{
			s_use_aspect_ratio_heuristics = use_aspect_ratio_heuristics_ex ? 2 : 1;
			reshade::config_set_value(nullptr, "DEPTH", "UseAspectRatioHeuristics", s_use_aspect_ratio_heuristics);
			force_reset = true;
		}
	}

	if (bool copy_before_clear_operations = s_preserve_depth_buffers != 0;
		ImGui::Checkbox("Copy depth buffer before clear operations", &copy_before_clear_operations))
	{
		s_preserve_depth_buffers = copy_before_clear_operations ? 1 : 0;
		reshade::config_set_value(nullptr, "DEPTH", "DepthCopyBeforeClears", s_preserve_depth_buffers);
		force_reset = true;
	}

	const bool is_d3d12_or_vulkan = device->get_api() == device_api::d3d12 || device->get_api() == device_api::vulkan;

	if (s_preserve_depth_buffers || is_d3d12_or_vulkan)
	{
		if (bool copy_before_fullscreen_draws = s_preserve_depth_buffers == 2;
			ImGui::Checkbox(is_d3d12_or_vulkan ? "Copy depth buffer during frame to prevent artifacts" : "Copy depth buffer before fullscreen draw calls", &copy_before_fullscreen_draws))
		{
			s_preserve_depth_buffers = copy_before_fullscreen_draws ? 2 : 1;
			reshade::config_set_value(nullptr, "DEPTH", "DepthCopyBeforeClears", s_preserve_depth_buffers);
		}
	}

	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Spacing();

	std::shared_lock<std::shared_mutex> lock(s_mutex);

	if (device_data.current_depth_stencil_list.empty())
	{
		ImGui::TextUnformatted("No depth buffers found.");
		return;
	}

	// Sort pointer list so that added/removed items do not change the GUI much
	struct depth_stencil_item
	{
		unsigned int display_count;
		resource resource;
		depth_stencil_info snapshot;
		resource_desc desc;
	};

	std::vector<depth_stencil_item> sorted_item_list;
	sorted_item_list.reserve(device_data.current_depth_stencil_list.size());

	for (const auto &[resource, snapshot] : device_data.current_depth_stencil_list)
	{
		if (auto it = data.display_count_per_depth_stencil.find(resource);
			it == data.display_count_per_depth_stencil.end())
		{
			sorted_item_list.push_back({ 1u, resource, snapshot, device->get_resource_desc(resource) });
		}
		else
		{
			sorted_item_list.push_back({ it->second + 1u, resource, snapshot, device->get_resource_desc(resource) });
		}
	}

	lock.unlock();

	std::sort(sorted_item_list.begin(), sorted_item_list.end(), [](const depth_stencil_item &a, const depth_stencil_item &b) {
		return (a.display_count > b.display_count) ||
			(a.display_count == b.display_count && ((a.desc.texture.width > b.desc.texture.width || (a.desc.texture.width == b.desc.texture.width && a.desc.texture.height > b.desc.texture.height)) ||
			(a.desc.texture.width == b.desc.texture.width && a.desc.texture.height == b.desc.texture.height && a.resource < b.resource)));
	});

	bool has_msaa_depth_stencil = false;
	bool has_no_clear_operations = false;

	data.display_count_per_depth_stencil.clear();
	for (const depth_stencil_item &item : sorted_item_list)
	{
		data.display_count_per_depth_stencil[item.resource] = item.display_count;

		char label[512] = "";
		sprintf_s(label, "%c 0x%016llx", (item.resource == data.selected_depth_stencil ? '>' : ' '), item.resource.handle);

		if (item.desc.texture.samples > 1) // Disable widget for MSAA textures
		{
			has_msaa_depth_stencil = true;

			ImGui::BeginDisabled();
			ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
		}

		if (bool value = (item.resource == data.override_depth_stencil);
			ImGui::Checkbox(label, &value))
		{
			data.override_depth_stencil = value ? item.resource : resource{ 0 };
			force_reset = true;
		}

		ImGui::SameLine();
		ImGui::Text("| %4ux%-4u | %s | %5u draw calls (%5u indirect) ==> %8u vertices |%s",
			item.desc.texture.width,
			item.desc.texture.height,
			format_to_string(item.desc.texture.format),
			item.snapshot.total_stats.drawcalls,
			item.snapshot.total_stats.drawcalls_indirect,
			item.snapshot.total_stats.vertices,
			(item.desc.texture.samples > 1 ? " MSAA" : ""));

		if (item.desc.texture.samples > 1)
		{
			ImGui::PopStyleColor();
			ImGui::EndDisabled();
		}

		if (s_preserve_depth_buffers && item.resource == data.selected_depth_stencil)
		{
			if (item.snapshot.clears.empty())
			{
				has_no_clear_operations = !is_d3d12_or_vulkan;
				continue;
			}

			depth_stencil_backup *const depth_stencil_backup = device_data.find_depth_stencil_backup(item.resource);
			if (depth_stencil_backup == nullptr || depth_stencil_backup->backup_texture == 0)
				continue;

			for (size_t clear_index = 1; clear_index <= item.snapshot.clears.size(); ++clear_index)
			{
				const auto &clear_stats = item.snapshot.clears[clear_index - 1];

				sprintf_s(label, "%c   CLEAR %2zu", clear_stats.copied_during_frame ? '>' : ' ', clear_index);

				if (bool value = (depth_stencil_backup->force_clear_index == clear_index);
					ImGui::Checkbox(label, &value))
				{
					depth_stencil_backup->force_clear_index = value ? clear_index : 0;
					reshade::config_set_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", depth_stencil_backup->force_clear_index);
				}

				ImGui::SameLine();
				ImGui::Text("        |           |       | %5u draw calls (%5u indirect) ==> %8u vertices |%s",
					clear_stats.drawcalls,
					clear_stats.drawcalls_indirect,
					clear_stats.vertices,
					clear_stats.clear_op == clear_op::fullscreen_draw ? " Fullscreen draw call" : "");
			}

			if (sorted_item_list.size() == 1 && !is_d3d12_or_vulkan)
			{
				if (bool value = (depth_stencil_backup->force_clear_index == std::numeric_limits<size_t>::max());
					ImGui::Checkbox("    Choose last clear operation with high number of draw calls", &value))
				{
					depth_stencil_backup->force_clear_index = value ? std::numeric_limits<size_t>::max() : 0;
					reshade::config_set_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", depth_stencil_backup->force_clear_index);
				}
			}
		}
	}

	if (has_msaa_depth_stencil || has_no_clear_operations)
	{
		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();

		ImGui::PushTextWrapPos();
		if (has_msaa_depth_stencil)
			ImGui::TextUnformatted("Not all depth buffers are available.\nYou may have to disable MSAA in the game settings for depth buffer detection to work!");
		if (has_no_clear_operations)
			ImGui::Text("No clear operations were found for the selected depth buffer.\n%s",
				s_preserve_depth_buffers != 2 ? "Try enabling \"Copy depth buffer before fullscreen draw calls\" or disable \"Copy depth buffer before clear operations\"!" : "Disable \"Copy depth buffer before clear operations\" or select a different depth buffer!");
		ImGui::PopTextWrapPos();
	}

	if (force_reset)
	{
		// Reset selected depth-stencil to force re-creation of resources next frame (like the backup texture)
		if (data.selected_shader_resource != 0)
		{
			command_queue *const queue = runtime->get_command_queue();

			queue->wait_idle(); // Ensure resource view is no longer in-use before destroying it
			device->destroy_resource_view(data.selected_shader_resource);

			device_data.untrack_depth_stencil(data.selected_depth_stencil);
		}

		data.using_backup_texture = false;
		data.selected_depth_stencil = { 0 };
		data.selected_shader_resource = { 0 };

		update_effect_runtime(runtime);
	}
}

void register_addon_depth()
{
	reshade::register_overlay(nullptr, draw_settings_overlay);

	reshade::register_event<reshade::addon_event::init_device>(on_init_device);
	reshade::register_event<reshade::addon_event::init_command_list>(on_init_command_list);
	reshade::register_event<reshade::addon_event::init_command_queue>(on_init_command_queue);
	reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
	reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
	reshade::register_event<reshade::addon_event::destroy_command_list>(on_destroy_command_list);
	reshade::register_event<reshade::addon_event::destroy_command_queue>(on_destroy_command_queue);
	reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);

	reshade::register_event<reshade::addon_event::create_resource>(on_create_resource);
	reshade::register_event<reshade::addon_event::create_resource_view>(on_create_resource_view);
	reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy_resource);

	reshade::register_event<reshade::addon_event::draw>(on_draw);
	reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
	reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(on_draw_indirect);
	reshade::register_event<reshade::addon_event::bind_viewports>(on_bind_viewport);
	reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_render_pass_with_depth_stencil);
	reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_depth_stencil);
	reshade::register_event<reshade::addon_event::clear_depth_stencil_view>(on_clear_depth_stencil);

	reshade::register_event<reshade::addon_event::reset_command_list>(on_reset);
	reshade::register_event<reshade::addon_event::execute_command_list>(on_execute_primary);
	reshade::register_event<reshade::addon_event::execute_secondary_command_list>(on_execute_secondary);

	reshade::register_event<reshade::addon_event::present>(on_present);

	reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_render_effects);
	reshade::register_event<reshade::addon_event::reshade_finish_effects>(on_finish_render_effects);
	// Need to set texture binding again after reloading
	reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(update_effect_runtime);
}
void unregister_addon_depth()
{
	reshade::unregister_event<reshade::addon_event::init_device>(on_init_device);
	reshade::unregister_event<reshade::addon_event::init_command_list>(on_init_command_list);
	reshade::unregister_event<reshade::addon_event::init_command_queue>(on_init_command_queue);
	reshade::unregister_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
	reshade::unregister_event<reshade::addon_event::destroy_device>(on_destroy_device);
	reshade::unregister_event<reshade::addon_event::destroy_command_list>(on_destroy_command_list);
	reshade::unregister_event<reshade::addon_event::destroy_command_queue>(on_destroy_command_queue);
	reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);

	reshade::unregister_event<reshade::addon_event::create_resource>(on_create_resource);
	reshade::unregister_event<reshade::addon_event::create_resource_view>(on_create_resource_view);
	reshade::unregister_event<reshade::addon_event::destroy_resource>(on_destroy_resource);

	reshade::unregister_event<reshade::addon_event::draw>(on_draw);
	reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
	reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(on_draw_indirect);
	reshade::unregister_event<reshade::addon_event::bind_viewports>(on_bind_viewport);
	reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_render_pass_with_depth_stencil);
	reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_depth_stencil);
	reshade::unregister_event<reshade::addon_event::clear_depth_stencil_view>(on_clear_depth_stencil);

	reshade::unregister_event<reshade::addon_event::reset_command_list>(on_reset);
	reshade::unregister_event<reshade::addon_event::execute_command_list>(on_execute_primary);
	reshade::unregister_event<reshade::addon_event::execute_secondary_command_list>(on_execute_secondary);

	reshade::unregister_event<reshade::addon_event::present>(on_present);

	reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(on_begin_render_effects);
	reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(on_finish_render_effects);
	reshade::unregister_event<reshade::addon_event::reshade_reloaded_effects>(update_effect_runtime);
}

extern "C" __declspec(dllexport) const char *NAME = "Citra";
extern "C" __declspec(dllexport) const char *DESCRIPTION = "add-on that pre-processes depth buffer from Citra to be standardized / aligned for other add-ons to consume it.";

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
		if (!reshade::register_addon(hinstDLL))
			return FALSE;
		register_addon_depth();
		break;
	case DLL_PROCESS_DETACH:
		unregister_addon_depth();
		reshade::unregister_addon(hinstDLL);
		break;
	}
	return TRUE;
}
