/*
  CitraLKG by Holophone3D, Jake Downs.

A very much WIP(Work In progress) attempt to turn Stereo3D into viewable Hologram on LookingGlass Portrait.

Reshade:
-Configure CitraAddOn with Dual Depth (no other addons enabled)
-Ensure that L and R depth buffers are correctly assigned to L and R respectively use Quad rendering mode to verify
-Add this file to your shaders folder to reshade-shaders/Shaders for reshade under Citra
-Add mario_depth_disparity_lut.png to reshade-shaders/Textures for reshade under Citra
-Enable CitraLKG.fx

Citra:
-3D set to 100%
-Change to SBS
-Set Internal Resolution to the highest value that allows you to run at 60+fps, I recommend at least 3x maybe 5x or more
-Ensure window is resized to remove ALL black from between the L and R images (I'm doing the hack to get the LR frames from the backbuffer)

In Game (SuperMario3DLand):
-Load save state for first mario level start
   - If you see a red square in the bottom right corner and/or the screen looks scrambled
   - Then depth buffer data did not bind correctly from Citra addon, you'll need to reload Save state
   - Sometimes it takes a few times for depth buffers to bind, so just keep trying
-Ensure 3D mode is 'inwards'

CitraLKG.fx:
-Update the LKG config settings for your device in .fx or in UI

Viewing on LKG:
-Set Render mode to 'LKG'
-Drag Citra window over to Looking Glass (mine is the left display of an extended desktop on Windows)
-Hit F11 to toggle full screen
-Running in 'Performance Mode' is recommended if you intend to play on the LKG

All Defaults are optimized for 100% 3D with '3D' in mode

Right now most of Level 1-1 in SuperMario3DLand looks ok. There's lots of issues outside of that.
Most of them should be solvable, but will just take some time.

*/

#include "ReShade.fxh"

// -- Textures --
// input
texture OrigDepthTex : ORIG_DEPTH;
sampler DepthLeft{
	Texture = OrigDepthTex;
	MipFilter = NONE; // Disable mipmapping
	AddressU = WRAP;
	AddressV = WRAP;
	AddressW = WRAP; };

texture OrigDepthTex2 : ORIG_DEPTH_2;
sampler DepthRight{
	Texture = OrigDepthTex2;
	MipFilter = NONE; // Disable mipmapping
	AddressU = WRAP;
	AddressV = WRAP;
	AddressW = WRAP; };

//TODO: Expose these textures in Citra Addon
texture TexRGBLeft : RGB_LEFT;
sampler RGBLeft{
	Texture = TexRGBLeft;
	MipFilter = NONE; // Disable mipmapping
	AddressU = WRAP;
	AddressV = WRAP;
	AddressW = WRAP; };

texture TexRGBRight : RGB_RIGHT;
sampler RGBRight{
	Texture = TexRGBRight;
	MipFilter = NONE; // Disable mipmapping
	AddressU = WRAP;
	AddressV = WRAP;
	AddressW = WRAP;
};

uniform int DEPTH_HISTOGRAM_BINS_NUM = 256;
uniform int DEPTH_HISTOGRAM_SMOOTHNESS = 1;
uniform int DEPTH_HISTOGRAM_SAMPLES_X = 400;
uniform int DEPTH_HISTOGRAM_SAMPLES_Y = 240;


// DEPTH histograms
texture HistogramTex{ Width = 256;   Height = 1; };
sampler sHistogramTex{ Texture = HistogramTex; };

// DEPTH slice far images
texture DepthSliceTex{ Width = 400;   Height = 240; };
sampler sDepthSliceTex{ Texture = DepthSliceTex; };

// DEPTH Transform 
texture DepthTransformTex{ Width = 1;   Height = 1; };
sampler sDepthTransformTex{ Texture = DepthTransformTex; };


// -- Options --
uniform int iUIRenderMode <
	ui_type = "combo";
ui_category = "Render Mode";
ui_label = "Mode";
ui_items = "Off\0" // 0
"Lightfield\0" //1
"Quilt\0" //2
"LKG\0" // 3
"Quad(Debug)\0" // 4
"UIMask(Debug)\0"; // 5
> = 0;
uniform int iUIScreenMode <
	ui_type = "combo";
ui_category = "Render Mode";
ui_label = "Screen Size";
ui_items = "Full(Stretched)\0" // 0
"LetterBox(Native)\0"; // 1
> = 0;
uniform float fUIDepthPercent <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "Depth%";
ui_min = 0; ui_max = 256;
ui_step = 1;
> = 100;

uniform float fUIOffset <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "Offset";
ui_min = -512; ui_max = 512;
ui_step = 0.5;
> = 0;
uniform float fUIAdaptiveOffsetRange <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "Adaptive Offset Range";
ui_min = 0; ui_max = 6;
ui_step = 1;
> = 2;
uniform bool fUIUXExtract <
	ui_category = "Render Mode";
ui_label = "UX Extract";
> = true;
uniform float fUIUXOffset <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "UX Offset";
ui_min = -100; ui_max = 100;
ui_step = 0.5;
> = -2.0;
uniform float fUIUXThreshold <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "UX Threshold";
ui_min = 0; ui_max = 3;
ui_step = 0.01;
> = 0;
uniform float fUIUXShiftThreshold <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "UX Shift Threshold";
ui_min = 0; ui_max = 3;
ui_step = 0.01;
> = 0;
uniform float fUIDepthTrimLeft <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "Depth Trim Left(pixels)";
ui_min = 0; ui_max = 200;
ui_step = 1;
> = 0;
uniform float fUIDepthTrimTop <
	ui_type = "drag";
ui_category = "Render Mode";
ui_label = "Depth Trim Top(pixels)";
ui_min = 0; ui_max = 200;
ui_step = 1;
> = 0;

uniform float fUIAlpha <
	ui_type = "drag";
ui_category = "Lightfield Options";
ui_label = "Alpha";
ui_min = 0.0; ui_max = 1.0;
ui_step = 0.01;
> = 0.5;

uniform float fUIAlphaExtended <
	ui_type = "drag";
ui_category = "Lightfield Options";
ui_label = "Extended Alpha";
ui_min = 0.0; ui_max = 1.0;
ui_step = 0.01;
> = 0.3;

uniform float fUIDepthiness <
	ui_type = "drag";
ui_category = "Quilt/LKG Options";
ui_label = "Depthiness";
ui_min = 0.0; ui_max = 2.0;
ui_step = 0.01;
> = 1.67;
uniform float fUIFocus <
	ui_type = "drag";
ui_category = "Quilt/LKG Options";
ui_label = "Focus";
ui_min = -1.0; ui_max = 1.0;
ui_step = 0.001;
> = 0.102;
uniform float fUIPadding <
	ui_type = "drag";
ui_category = "Quilt/LKG Options";
ui_label = "Padding";
ui_min = 0.0; ui_max = 0.5;
ui_step = 0.01;
> = 0.02;

uniform float fUIClip <
	ui_type = "drag";
ui_category = "UI Mask Options";
ui_label = "UI Clip";
ui_min = 0.0; ui_max = 1.0;
ui_step = 0.01;
> = 0.06;

uniform int iUIDepthDebuggerMode <
	ui_type = "combo";
ui_category = "Depth Debugger";
ui_label = "Debug Mode";
ui_items = "Off\0" // 0
"Color\0" //1
"ColorDelta\0" //2
"Depth\0" //3
"Normals\0"; // 4
> = 0;
uniform float fUIDepthLayer <
	ui_type = "drag";
ui_category = "Depth Debugger";
ui_label = "Depth Layer";
ui_min = 0; ui_max = 256;
ui_step = 1;
> = 230;

//Interactive depth controls
uniform bool overlay_open < source = "overlay_open"; > ;
uniform bool left_mouse_button_down < source = "mousebutton"; keycode = 0; mode = ""; > ;
uniform float2 mouse_point < source = "mousepoint"; > ;
uniform bool left_ctrl_down < source = "key"; keycode = 0x11; mode = ""; > ;
uniform bool space_bar_down < source = "key"; keycode = 0x20; mode = ""; > ;

// https://jakedowns.github.io/looking-glass-calibration.html
// LKG calibration data  (put your LKG data here)
uniform float width = 1536.0f;
uniform float height = 2048.0f;
uniform float dpi = 324.0f;
uniform float slope = -7.198658892370121;
uniform float center = 0.5990424947346542;
uniform float pitch = 52.57350593130944;
uniform int invView = 1;
uniform float displayAspect = 0.75;
uniform int ri = 0;
uniform int bi = 2;

// Quilt data (defaults)
uniform float2 quilt_tile = float2(11.0, 9.0); //cols, rows, total views (will be computed)
uniform int overscan = 0;
uniform int quiltInvert = 0;

// match glsl mod behavior
float mod(float x, float y) {
	return x - y * floor(x / y);
}

float normalizeDepth(float depth)
{
	float minDepth = 100.0; // closest report depth value in the depth buffer 
	float originalMin = minDepth / 256.0;
	float originalMax = 1.0;
	return lerp(0.0, 1.0, (depth - originalMin) / (originalMax - originalMin));
}

int quantizeDepth(float depth)
{
	return floor(depth * 256);
}

float3 ConvertToGrayscale(float3 color)
{
	float grayscale = dot(color, float3(0.299, 0.587, 0.114));
	return float3(grayscale, grayscale, grayscale);
}

// Gets depth from Citra depth buffers and returns in RGB aligned space
float GetModDepth(sampler depth_tex, float2 tex : TEXCOORD, bool normalize) {

	float left_trim = fUIDepthTrimLeft * BUFFER_PIXEL_SIZE.x;
	float top_trim = fUIDepthTrimTop * BUFFER_PIXEL_SIZE.y;

	// Remap the X coordinate to the range of the specified image
	tex.x = lerp(left_trim, 1.0, tex.x);
	// Remap the Y coordinate to the range of the specified image
	tex.y = lerp(top_trim, 1.0, tex.y);

	float depth = tex2Dlod(depth_tex, float4(1.0 - float2(tex.y, tex.x), 0, 0)).x;
	if (normalize)
		depth = normalizeDepth(depth);
	return depth;
}

float3 GetScreenSpaceNormal(sampler depth_tex, float2 texcoord)
{
	float3 offset = float3(BUFFER_PIXEL_SIZE, 0.0);
	float2 posCenter = texcoord.xy;
	float2 posNorth = posCenter - offset.zy;
	float2 posEast = posCenter + offset.xz;

	float3 vertCenter = float3(posCenter - 0.5, 1) * GetModDepth(depth_tex, posCenter, false);
	float3 vertNorth = float3(posNorth - 0.5, 1) * GetModDepth(depth_tex, posNorth, false);
	float3 vertEast = float3(posEast - 0.5, 1) * GetModDepth(depth_tex, posEast, false);

	return normalize(cross(vertCenter - vertNorth, vertCenter - vertEast)) * 0.5 + 0.5;
}

// extracts clean L and R images from the SxS backbuffer no matter what size the base window is
// still not perfect for all layouts, but pretty close and works well with LKG
float2 RemapBackBufferToImage(float2 pos, int side)
{
	// Define the aspect ratio
	static const float ASPECT_RATIO = 1.2;

	// Calculate the aspect ratio of the buffer
	float bufferAspectRatio = BUFFER_WIDTH / BUFFER_HEIGHT;

	float2 remappedPos = pos;

	if (bufferAspectRatio > ASPECT_RATIO)
	{
		// Buffer is wider than the target aspect ratio
		float imageWidth = BUFFER_HEIGHT / ASPECT_RATIO;
		float imageHeight = BUFFER_HEIGHT;

		// ensure with is even so we don't have odd or fractional pixels
		imageWidth = (imageWidth % 2) == 0 ? imageWidth : imageWidth - 1;

		// Normalize imageWidth and imageHeight between 0 and 1
		imageWidth *= BUFFER_PIXEL_SIZE.x;
		imageHeight *= BUFFER_PIXEL_SIZE.y;

		// Calculate the starting X position based on the side parameter
		float startX = (side == 1) ? 0.25 - (imageWidth / 2.0) : 0.75 - (imageWidth / 2.0);

		// Remap the X coordinate to the range of the specified image
		remappedPos.x = lerp(startX, startX + imageWidth, remappedPos.x);

		// Remap the Y coordinate to the range of the images
		remappedPos.y = lerp(0.5 - imageHeight / 2.0, 0.5 + imageHeight / 2.0, remappedPos.y);
	}
	else
	{
		// Buffer is taller than the target aspect ratio
		float imageWidth = BUFFER_WIDTH / 2.0;
		float imageHeight = imageWidth * ASPECT_RATIO;

		// ensure with is even so we don't have odd or fractional pixels
		imageWidth = (imageWidth % 2) == 0 ? imageWidth : imageWidth - 1;

		// Normalize imageWidth and imageHeight between 0 and 1
		imageWidth *= BUFFER_PIXEL_SIZE.x;
		imageHeight *= BUFFER_PIXEL_SIZE.y;

		// Calculate the starting Y position based on the side parameter
		float startY = 0.5 - (imageHeight / 2.0);

		// Remap the Y coordinate to the range of the specified image
		remappedPos.y = lerp(startY, startY + imageHeight, remappedPos.y);

		// Offset the X coordinate based on the side parameter
		float startX = (side == 1) ? 0.0 : 0.5;

		// Remap the X coordinate to the range of the specified image
		remappedPos.x = lerp(startX, startX + imageWidth, remappedPos.x);
	}

	return remappedPos;
}

float4 getLeftRGB(float2 normalized_coords)
{
	return tex2D(ReShade::BackBuffer, RemapBackBufferToImage(normalized_coords, 1));
}

float4 getRightRGB(float2 normalized_coords)
{
	return tex2D(ReShade::BackBuffer, RemapBackBufferToImage(normalized_coords, 2));
}

float4 getLeftDepth(float2 normalized_coords, bool normalize)
{
	float d = GetModDepth(DepthLeft, normalized_coords, normalize);
	return float4(d, d, d, 1.0);
}

float4 getRightDepth(float2 normalized_coords, bool normalize)
{
	float d = GetModDepth(DepthRight, normalized_coords, normalize);
	return float4(d, d, d, 1.0);
}

float4 getLeftNormal(float2 normalized_coords)
{
	return float4(GetScreenSpaceNormal(DepthLeft, normalized_coords), 1.0);
}

float4 getRightNormal(float2 normalized_coords)
{
	return float4(GetScreenSpaceNormal(DepthRight, normalized_coords), 1.0);
}

// Determines how far to shift a pixel based on depth information
float get_depth_shift(float2 normalized_coords, float depth_value, float offset_adjustment)
{
	//TODO: Working formula for depth at 100% - need to compute this adaptively
	float offset = (1.181 + (-0.909 / depth_value)) * ((fUIDepthPercent / 100.0) * 0.325);
	offset += offset_adjustment / 256.0;
	return offset;
}

float cosSim(float3 x, float3 y)
{
	return dot(x, y) / (length(x) * length(y));
}

float4 getShiftedColorDepthInfo(float2 normalized_coords, int x_shift, in float alpha, out float3 left_rgb, out float3 right_rgb, out float depth_l, out float depth_r, out float depth_shifted_x_l, out float depth_shifted_x_r, out float depth_shifted_alpha)
{
	float4 color = float4(1.0, 1.0, 1.0, 1.0);

	depth_l = GetModDepth(DepthLeft, normalized_coords, false);
	depth_r = GetModDepth(DepthRight, normalized_coords, false);

	float2 Shift_L = float2(get_depth_shift(normalized_coords, depth_l, x_shift), 0.0);
	float2 Shift_R = float2(get_depth_shift(normalized_coords, depth_r, x_shift), 0.0);

	depth_l = GetModDepth(DepthLeft, normalized_coords - alpha * Shift_L, false);
	depth_r = GetModDepth(DepthRight, normalized_coords + (1.0 - alpha) * Shift_R, false);

	// compute normalized shifted position
	depth_shifted_x_l = clamp(normalized_coords.x - alpha * get_depth_shift(normalized_coords, depth_l, x_shift), 0.0, 1.0);
	depth_shifted_x_r = clamp(normalized_coords.x + (1.0 - alpha) * get_depth_shift(normalized_coords, depth_r, x_shift), 0.0, 1.0);

	// Handle warping beyond known pixel data for single images and both images screen edge occlusion
	// we've exceeded pixel data for one image, so only use the data from the image still in bounds
	if (depth_shifted_x_l <= 0.001)
	{
		alpha = 1.0;
		depth_shifted_x_l = 0.001;
	}
	else if (depth_shifted_x_r == 1.0)
		alpha = 0.0;

	// TODO: figure out why 0.0 is grabbing bits from elsewhere
	if (depth_shifted_x_r <= 0.001)
	{
		alpha = 1.0;
		depth_shifted_x_r = 0.001;
	}
	else if (depth_shifted_x_l == 1.0)
		alpha = 0.0;

	// clamp color_mix_alpha so as not to over saturate colors below when using extended alpha
	depth_shifted_alpha = clamp(alpha, 0.0, 1.0);
	/*
		if (depth_shifted_alpha < 0.75 && depth_shifted_alpha > 0.25)
		{
			// test ensure that background pixels don't occlude closer pixels
			if(depth_shifted_x_l > depth_shifted_x_r)
				depth_shifted_alpha = 0.0;
			else if(depth_shifted_x_l < depth_shifted_x_r)
				depth_shifted_alpha = 1.0;
		}

		*/
		// sample alpha mixed pixels at shifted positions
	left_rgb = (1.0 - depth_shifted_alpha) * getLeftRGB(float2(depth_shifted_x_l, normalized_coords.y)).xyz;
	right_rgb = depth_shifted_alpha * getRightRGB(float2(depth_shifted_x_r, normalized_coords.y)).xyz;

	color = float4(left_rgb + right_rgb, 1.0);

	return color;
}

float4 get_Im(float2 normalized_coords, float alpha)
{

	float3 left_rgb;
	float3 right_rgb;
	float depth_l;
	float depth_r;
	float depth_shifted_x_l;
	float depth_shifted_x_r;
	float depth_shifted_alpha;

	float4 color = getShiftedColorDepthInfo(normalized_coords, fUIOffset, alpha, left_rgb, right_rgb, depth_l, depth_r, depth_shifted_x_l, depth_shifted_x_r, depth_shifted_alpha);

	if (fUIUXExtract)
	{
		// test to see if the pixel matches better at the UX target layer, it might be UX so dont depth shift it

		float3 shifted_left_rgb = getLeftRGB(float2(depth_shifted_x_l, normalized_coords.y)).xyz;
		float3 shifted_right_rgb = getRightRGB(float2(depth_shifted_x_r, normalized_coords.y)).xyz;
		float3 shifted_delta = abs(shifted_left_rgb - shifted_right_rgb);

		float Shift_L = fUIUXOffset * BUFFER_PIXEL_SIZE.x;
		float Shift_R = -fUIUXOffset * BUFFER_PIXEL_SIZE.x;

		float3 ux_left_rgb = getLeftRGB(float2(Shift_L, normalized_coords.y)).xyz;
		float3 ux_right_rgb = getRightRGB(float2(Shift_R, normalized_coords.y)).xyz;
		float3 ux_delta = abs(ux_left_rgb - ux_right_rgb);
		if (length(ux_delta) < fUIUXThreshold && length(shifted_delta) > fUIUXShiftThreshold)
		{

			depth_l = 230;
			depth_r = 230;

			// compute normalized shifted position
			depth_shifted_x_l = clamp(normalized_coords.x - alpha * get_depth_shift(normalized_coords, depth_l, fUIOffset), 0.0, 1.0);
			depth_shifted_x_r = clamp(normalized_coords.x + (1.0 - alpha) * get_depth_shift(normalized_coords, depth_r, fUIOffset), 0.0, 1.0);

			// Handle warping beyond known pixel data for single images and both images screen edge occlusion
			// we've exceeded pixel data for one image, so only use the data from the image still in bounds
			if (depth_shifted_x_l <= 0.001)
			{
				alpha = 1.0;
				depth_shifted_x_l = 0.001;
			}
			else if (depth_shifted_x_r == 1.0)
				alpha = 0.0;

			// TODO: figure out why 0.0 is grabbing bits from elsewhere
			if (depth_shifted_x_r <= 0.001)
			{
				alpha = 1.0;
				depth_shifted_x_r = 0.001;
			}
			else if (depth_shifted_x_l == 1.0)
				alpha = 0.0;

			// clamp color_mix_alpha so as not to over saturate colors below when using extended alpha
			depth_shifted_alpha = clamp(alpha, 0.0, 1.0);
			/*
				if (depth_shifted_alpha < 0.75 && depth_shifted_alpha > 0.25)
				{
					// test ensure that background pixels don't occlude closer pixels
					if(depth_shifted_x_l > depth_shifted_x_r)
						depth_shifted_alpha = 0.0;
					else if(depth_shifted_x_l < depth_shifted_x_r)
						depth_shifted_alpha = 1.0;
				}

				*/
				// sample alpha mixed pixels at shifted positions
			left_rgb = (1.0 - depth_shifted_alpha) * getLeftRGB(float2(depth_shifted_x_l, normalized_coords.y)).xyz;
			right_rgb = depth_shifted_alpha * getRightRGB(float2(depth_shifted_x_r, normalized_coords.y)).xyz;

			color = float4(left_rgb + right_rgb, 1.0);

		}
	}

	// adaptive offset
	//TODO: offset is getting weird at more alpha levels
	if (alpha >= 0.005 || alpha <= 0.995)
	{
		float best_xl = depth_shifted_x_l;
		float best_xr = depth_shifted_x_r;
		float4 best_color = color;
		float3 best_pixel_delta = left_rgb - right_rgb;
		float best_cs = cosSim(left_rgb, right_rgb);

		// find best offset to account for adaptive camera angles
		int range = fUIAdaptiveOffsetRange;
		for (int i = range; i >= -range; i--)
		{
			float x_shift = (i * 0.5) + fUIOffset;

			float4 test_color = getShiftedColorDepthInfo(normalized_coords, x_shift, alpha, left_rgb, right_rgb, depth_l, depth_r, depth_shifted_x_l, depth_shifted_x_r, depth_shifted_alpha);

			float3 delta = left_rgb - right_rgb;
			float cs = cosSim(left_rgb, right_rgb);
			if (i != 0 && length(delta) <= length(best_pixel_delta))
			{
				best_xl = depth_shifted_x_l;
				best_xr = depth_shifted_x_r;
				best_pixel_delta = delta;
				best_cs = cs;
				best_color = test_color;
				if (length(delta) < 0.01)
					break;
			}

		}

		color = best_color;
		depth_shifted_x_l = best_xl;
		depth_shifted_x_r = best_xr;
	}


	if (iUIDepthDebuggerMode != 0)
	{
		bool emphasizeActivePixel = false;
		// darken non active depth layers
		if ((quantizeDepth(depth_l) == fUIDepthLayer && quantizeDepth(depth_r) == fUIDepthLayer))
		{
			emphasizeActivePixel = true;
		}
		// Depth debugger visualization
		switch (iUIDepthDebuggerMode)
		{
		case 0: //OFF
			break;
		case 1: //color
			break;
		case 2: //color delta
			color = float4(abs(right_rgb - left_rgb), 1.0);
			if (emphasizeActivePixel)
				color += float4(0.1, 0.1, 0.1, 1.0);
			break;
		case 3: //depth
			float ld = (1.0 - depth_shifted_alpha) * getLeftDepth(float2(depth_shifted_x_l, normalized_coords.y), true).x;
			float rd = depth_shifted_alpha * getRightDepth(float2(depth_shifted_x_r, normalized_coords.y), true).x;
			float d = ld + rd;
			color = float4(d, d, d, 1.0);
			break;
		case 4: //normals
			float3 ln = (1.0 - depth_shifted_alpha) * getLeftNormal(float2(depth_shifted_x_l, normalized_coords.y)).xyz;
			float3 rn = depth_shifted_alpha * getRightNormal(float2(depth_shifted_x_r, normalized_coords.y)).xyz;
			color = float4(ln + rn, 1.0);
			break;
		default:
			break;
		}
		if (!emphasizeActivePixel)
			color *= 0.5;

	}
	return color;

}

float4 sampleImage(float2 normalized_coords, float alpha, float alpha_range_offset)
{
	// Native Aspect ratio
	if (iUIScreenMode == 1)
	{
		// format to 3DS native aspect ration of 1.6667 400/240
		float y_clip = (BUFFER_HEIGHT - (BUFFER_HEIGHT / 1.66667)) / BUFFER_HEIGHT / 2.0;
		// remap y to native aspect ratio
		if (normalized_coords.y < (1.0 - y_clip) && normalized_coords.y > y_clip)
			normalized_coords.y = lerp(0.0, 1.0, (normalized_coords.y - y_clip) / ((1.0 - y_clip) - y_clip));
		// exit early and return black for any pixel outside of the aspect ratio
		else
			return float4(0.0, 0.0, 0.0, 1.0);
	}
	alpha = lerp(-alpha_range_offset, 1.0 + alpha_range_offset, alpha);
	return get_Im(normalized_coords, alpha);
}

float3 get_quilt_coords(float2 normalized_coords, float2 quilt_layout, float focus_val)
{
	// quilt config
	float cols = quilt_layout.x;
	float rows = quilt_layout.y;
	float view_count = rows * cols;

	// alter virtual camera angle for each quilt image
	float alpha_delta = 1.0 / (view_count - 1.0); // step per quilt view, top right needs to be at alpha 1.0 so subtract -1
	float active_row = floor((1.0 - normalized_coords.y) * rows); // invert y-axis calculation
	float active_col = floor(normalized_coords.x * cols);
	float alpha = active_row * cols * alpha_delta + active_col * alpha_delta; // update alpha virtual camera angle based on quilt view

	// QUILT FOCUS
	// map the view_num to -1 to 1 (0 is leftmost n is rightmost)
	float view_num = (active_row * cols + active_col);
	float focusAdjust = view_num / view_count * 2.0 - 1.0;
	// multiply by the focus value (normally a user accessible slider)
	focusAdjust *= focus_val;

	// update coordinates for quilt space
	float2 quilt_coords = float2(normalized_coords.x * cols, (1.0 - normalized_coords.y) * rows); // invert y-axis calculation
	quilt_coords -= float2(active_col, active_row); // subtract active_col and active_row
	quilt_coords += float2(focusAdjust, 0.0); // add focusAdjust only to x-coordinate
	quilt_coords.y = 1.0 - quilt_coords.y; // flip y back again
	// return the updated coord data and alpha
	return float3(quilt_coords, alpha);
}

float rescaleAlpha(float value, float padFactor)
{
	float lowerThreshold = padFactor;
	float upperThreshold = 1.0 - padFactor;

	if (value <= lowerThreshold)
	{
		return 0.0;
	}
	else if (value <= upperThreshold)
	{
		return (value - lowerThreshold) / (upperThreshold - lowerThreshold);
	}
	else
	{
		return 1.0;
	}
}

float4 sampleVirtualQuiltImage(float2 normalized_coords, float alpha_range_offset, float focus_val, float padding_val)
{
	float3 quilt_data = get_quilt_coords(normalized_coords, quilt_tile, focus_val);
	float alpha = rescaleAlpha(quilt_data.z, padding_val);
	return sampleImage(quilt_data.xy, alpha, alpha_range_offset);
}

float2 texArr(float3 uvz, float3 tile, float2 viewPortion)
{
	// decide which section to take from based on the z.
	float x = (mod(uvz.z, tile.x) + uvz.x) / tile.x;
	float y = (floor(uvz.z / tile.x) + (uvz.y)) / tile.y;
	return float2(x, -y) * viewPortion.xy;
}

void clip(float3 toclip)
{
	if (any(toclip < float3(0, 0, 0)))
		discard;
}

float4 renderToLookingGlass(float2 normalized_coords, float alpha_range_offset, float focus_val, float padding_val)
{
	// update computed LKG values from calibration data    
	float subp = 1.0 / (3.0 * width);
	float tilt = height / (width * slope);
	float adjusted_pitch = pitch * (width / dpi) * sin(atan(abs(slope)));
	float3 tile = float3(quilt_tile.x, quilt_tile.y, quilt_tile.x * quilt_tile.y);
	float quiltAspect = (tile.y > tile.x) ? tile.x / tile.y : tile.y / tile.x;
	float2 viewPortion = float2(1.0, 1.0); //todo: still don't know if this is computed or not

	float invert = 1.0;
	if (invView + quiltInvert == 1)
		invert = -1.0;

	float3 nuv = float3(normalized_coords.x, 1.0 - normalized_coords.y, 0.0);
	nuv -= 0.5;
	float modx = clamp(step(quiltAspect, displayAspect) * step(float(overscan), 0.5) + step(displayAspect, quiltAspect) * step(0.5, float(overscan)), 0.0, 1.0);
	nuv.x = modx * nuv.x * displayAspect / quiltAspect + (1.0 - modx) * nuv.x;
	nuv.y = modx * (nuv.y) + (1.0 - modx) * (nuv.y) * quiltAspect / displayAspect;
	nuv += 0.5;
	clip(nuv);
	clip(1.0 - nuv);

	float4 rgb[3];
	for (int i = 0; i < 3; i++)
	{
		nuv.z = (normalized_coords.x + float(i) * subp + (1.0 - normalized_coords.y) * tilt) * adjusted_pitch - center;
		nuv.z = mod(nuv.z + ceil(abs(nuv.z)), 1.0);
		nuv.z *= invert;
		nuv.z *= tile.z;
		float3 coords1 = nuv;
		float3 coords2 = nuv;
		coords1.y = coords2.y = clamp(nuv.y, 0.005, 0.995);
		coords1.z = floor(nuv.z);
		coords2.z = ceil(nuv.z);
		float4 col1 = sampleVirtualQuiltImage(texArr(coords1, tile, viewPortion), alpha_range_offset, focus_val, padding_val);
		float4 col2 = sampleVirtualQuiltImage(texArr(coords2, tile, viewPortion), alpha_range_offset, focus_val, padding_val);
		rgb[i] = lerp(col1, col2, nuv.z - coords1.z);
	}
	return float4(rgb[ri].r, rgb[1].g, rgb[bi].b, 1.0);
}


float GetAlpha()
{
	if (left_mouse_button_down && !overlay_open)
		return clamp(mouse_point.x / BUFFER_WIDTH, 0.0, 1.0);
	else
		return fUIAlpha;
}

float GetDepthiness()
{
	if (left_mouse_button_down && !overlay_open && !left_ctrl_down)
		return 3.0 * clamp(mouse_point.x / BUFFER_WIDTH, 0.0, 1.0);
	else
		return fUIDepthiness;
}

float GetFocus()
{
	if (left_mouse_button_down && !overlay_open)
		return clamp(1.0 - (mouse_point.y / BUFFER_HEIGHT), 0.0, 1.0) / 2.0;
	else
		return fUIFocus;
}

float GetPadding()
{
	if (left_mouse_button_down && !overlay_open && left_ctrl_down)
		return 0.5 * clamp(mouse_point.x / BUFFER_WIDTH, 0.0, 1.0);
	else
		return fUIPadding;
}

float4 GenerateLightfield(float4 pos : SV_POSITION, float2 tex : TEXCOORD) : SV_TARGET{
	return sampleImage(tex, GetAlpha(), fUIAlphaExtended);
}

float4 GenerateQuilt(float4 pos : SV_POSITION, float2 tex : TEXCOORD) : SV_TARGET{
	return sampleVirtualQuiltImage(tex, GetDepthiness(), GetFocus(), GetPadding());
}

float4 GenerateLKG(float4 pos : SV_POSITION, float2 tex : TEXCOORD) : SV_TARGET{
	return renderToLookingGlass(tex, GetDepthiness(), GetFocus(), GetPadding());
}

float4 GenerateQuad(float4 pos : SV_POSITION, float2 tex : TEXCOORD) : SV_TARGET{
  if (tex.x < 0.5 && tex.y < 0.5)
	return getLeftRGB(tex * 2.0);
  else if (tex.x >= 0.5 && tex.y < 0.5)
	return getRightRGB(float2(tex.x - 0.5, tex.y) * 2.0);
  else if (tex.x < 0.5 && tex.y >= 0.5)
	return getLeftDepth(float2(tex.x,tex.y - 0.5) * 2.0, false);
  else if (tex.x >= 0.5 && tex.y >= 0.5)
	return getRightDepth((tex - 0.5) * 2.0, false);
  else
	return float4(1.0,0.0,0.0,1.0);
}

float4 GenerateUIMask(float4 pos : SV_POSITION, float2 tex : TEXCOORD) : SV_TARGET{

	float3 left_rgb;
	float3 right_rgb;
	float depth_l;
	float depth_r;
	float depth_shifted_x_l;
	float depth_shifted_x_r;
	float depth_shifted_alpha;

	float4 color = getShiftedColorDepthInfo(tex, fUIOffset, fUIAlpha, left_rgb,  right_rgb, depth_l, depth_r,  depth_shifted_x_l,  depth_shifted_x_r, depth_shifted_alpha);
	color = float4(abs(right_rgb - left_rgb), 1.0);
	if (ConvertToGrayscale(color.xyz).x > fUIClip)
		color = float4(1.0,1.0,1.0,1.0);
	else
		color = float4(0.0,0.0,0.0,0.0);

	return clamp(color, 0.0, 1.0);
}

void NotifyInvalidDepthBuffers(float2 tex, inout float4 color)
{
	if ((tex.x > 0.9 && tex.y > 0.9) && ((GetModDepth(DepthLeft, tex, false) <= 0.001) || (GetModDepth(DepthRight, tex, false) <= 0.001)))
	{
		color = float4(1.0, 0.0, 0.0, 1.0);
	}
}

float4 RenderImage(float4 pos : SV_POSITION, float2 tex : TEXCOORD) : SV_TARGET{
	int renderMode = iUIRenderMode;
// use spacebar to override rendermode to lightfield
if (space_bar_down && !overlay_open)
{
	renderMode = 1;
}

float4 color = float4(0.0,0.0,0.0,1.0);

switch (renderMode)
{
	case 0:
		color = tex2D(ReShade::BackBuffer, tex);
		break;
	case 1:
		color = GenerateLightfield(pos, tex);
		break;
	case 2:
		color = GenerateQuilt(pos, tex);
		break;
	case 3:
		color = GenerateLKG(pos, tex);
		break;
	case 4:
		color = GenerateQuad(pos, tex);
		break;
	case 5:
		color = GenerateUIMask(pos, tex);
		break;
	default:
		break;
}
color = tex2D(sDepthSliceTex, tex);

if (tex.y < 0.1)
	color = tex2D(sHistogramTex, float2(tex.x, 0)).xxx;
if (tex.y < 0.05)
	color = tex2D(sHistogramTex, float2(tex.x, 0)).yyy;
NotifyInvalidDepthBuffers(tex, color);
return color;
}

// modified version of histogram from martymcfly
// some stability issues here, seems worse on some parts of games and when not in performance mode
// https://github.com/martymcmodding/qUINT/blob/65fee6f2f00b9e86667dd986c0b196331dd193c2/Shaders/qUINT_lightroom.fx
// very expensive to compute both textures every frame
void PS_HistogramGenerate(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 res : SV_Target0)
{
	res = 0; float4 coord = 0;
	coord.z = rcp(DEPTH_HISTOGRAM_SAMPLES_X);
	coord.w = rcp(DEPTH_HISTOGRAM_SAMPLES_Y);

	float2 histogram_data = float2(DEPTH_HISTOGRAM_BINS_NUM, vpos.x) / DEPTH_HISTOGRAM_SMOOTHNESS;

	[loop]
	for (int x = 0; x < DEPTH_HISTOGRAM_SAMPLES_X; x++)
	{
		coord.y = 0;
		[loop]
		for (int y = 0; y < DEPTH_HISTOGRAM_SAMPLES_Y; y++)
		{
			res.x += saturate(1.0 - abs(getLeftDepth(coord.xy, false).x * histogram_data.x - histogram_data.y));
			res.y += saturate(1.0 - abs(getRightDepth(coord.xy, false).x * histogram_data.x - histogram_data.y));
			coord.y += coord.w;
		}
		coord.x += coord.z;
	}
	res.xy /= DEPTH_HISTOGRAM_SAMPLES_X * DEPTH_HISTOGRAM_SAMPLES_Y;
	res.xy *= 10.0;
}

float compressVector(float3 vec)
{
	return (vec.x + vec.y + vec.z) / 3.0;
}

void PS_DepthSliceGenerate(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float4 res : SV_Target0)
{
	int bins = 256;
	float step = 1.0 / DEPTH_HISTOGRAM_BINS_NUM;
	float prev_value = 0.0;
	float value = 0.0;
	float2 low_max = float2(0.0, 0.0);
	float2 high_max = float2(0.0, 0.0);
	float2 range = float2(0.0, 1.0);
	float2 peaks = float2(0.0, 1.0);
	float offset = 1.0 / bins;
	float2 max1 = float2(0.0, 0.0);
	float2 max2 = float2(0.0, 0.0);
	float2 max3 = float2(0.0, 0.0);
	float2 peak_curve_data = float2(0.0, 0.0);


	// align histograms
	float h_align = step;
	float2 best_h_delta = float2(bins, 0.0);
	[loop]
	for (int i = -3; i < 4; i++)
	{

		float histogram_delta = 0.0;
		[loop]
		for (int j = 0; j < bins; j++)
		{
			h_align = (1.0 * j) / bins;

			float l_d = tex2D(sHistogramTex, float2(h_align, 0)).x;
			float r_d = tex2D(sHistogramTex, float2(h_align + (i * step), 0)).y;
			histogram_delta += l_d - r_d;
		}
		if (histogram_delta < best_h_delta.x)
		{
			best_h_delta.x = histogram_delta;
			best_h_delta.y = i;
		}
	}


	//find histogram peaks
	[loop]
	for (int x = 0; x < bins; x++)
	{
		offset = (1.0 * x) / bins;

		value = tex2D(sHistogramTex, float2(offset, 0)).x;

		if (range.x == 0.0 && value != 0.0)
			range.x = offset;
		if (value != 0.0)
			range.y = offset;

		//detect peak - consider quantizing to reduce noise
		peak_curve_data.y = value - prev_value;

		if (peak_curve_data.x >= 0 && peak_curve_data.y < 0)
		{
			if (value > max3.x)
			{
				max1 = max2;
				max2 = max3;
				max3.x = value;
				max3.y = x;
			}
			else if (value > max2.x)
			{
				max1 = max2;
				max2.x = value;
				max2.y = x;
			}
			else if (value > max1.x)
			{
				max1.x = value;
				max1.y = x;
			}
		}
		// update for next interval
		prev_value = value;
		peak_curve_data.x = peak_curve_data.y;

	}

	// extract histogram peaks as new color channels
	// consider if I should create 4 render targets or just produce one image with all the info encoded in one image

/*	int r_depth = quantizeDepth(getRightDepth(uv.xy, false).x);
	if (depth == max1.y || depth == max2.y || depth == max3.y)
	{
		res = getRightRGB(uv.xy);
	}
	float v = (best_h_delta.y + 3.0) / 6.0;
//	res = float3(v,v,v);
*/

// Extract all color channel data into a single texture
	int r_depth = quantizeDepth(getRightDepth(uv.xy, false).x);
	if (r_depth == max1.y)
	{
		res.x = compressVector(getRightRGB(uv.xy).xyz);
	}
	else if (r_depth == max3.y)
	{
		res.y = compressVector(getRightRGB(uv.xy).xyz);
	}

	int l_depth = quantizeDepth(getLeftDepth(uv.xy, false).x);
	if (l_depth == max1.y)
	{
		res.z = compressVector(getLeftRGB(uv.xy).xyz);
	}
	else if (l_depth == max3.y)
	{
		res.w = compressVector(getLeftRGB(uv.xy).xyz);
	}
}



technique CitraLKG {
	//TODO: Always render the synthetic : image, depth and normals to texures
	// render modes are mainly overrides?
	// alternatively render offset maps, that contain the x/y pos in the LR images for lookup - maybe use xy for L and zw for R? - ideally float backed for precision, not int backed

	pass PHistogramGenerate
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_HistogramGenerate;
		RenderTarget = HistogramTex;
	}
	// align histograms pass
	// render best depth values
	// sDepthSliceNearRTex

	pass DepthSliceGenerate
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_DepthSliceGenerate;
		RenderTarget = DepthSliceTex;
	}

	pass GenerateImage
	{
		VertexShader = PostProcessVS;
		PixelShader = RenderImage;
	}
}