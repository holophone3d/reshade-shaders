//!HOOK OUTPUT
//!BIND HOOKED
//!DESC LookingGlass
// ! WIDTH 1536
// ! HEIGHT 2048

// Calibration values 
// via: https://jakedowns.github.io/looking-glass-calibration.html
float slope = -7.083540916442871;
float center = 0.8167931437492371;
float pitch = 52.59267044067383;

// this is especially made for looking glass portrait.
// TODO: make it more generic / able to support other variations
float width = 1536.0f; //HOOKED_size.x; //1536.0f;
// float width = HOOKED_size.x;
float height = 2048.0f; //1536.0f * 1920.0f / 2048.0f; 
// float height = HOOKED_size.y; //2048.0f;
// float height = width * 0.563;
float dpi = 324.0f;
// float dpi = 72.0f;

float tilt = height / (width * slope);
float pitch_adjusted = pitch * width / dpi * cos(atan(1.0f, slope));
// float tilt = -0.12039111681976107; //{tilt};
// float pitch = 370.66407267416486; //{pitch};
// float center = 0.13695651292800903 + tilt*pitch; //{center};
float subp = 1.0f / (3.0f * width) * pitch_adjusted;
float repeat = 100.0f/3.0f;

vec4 my_sample(float alpha){
	vec2 pos = HOOKED_pos;
	float halfX = pos.x / 2.0f;

	// return HOOKED_texOff(vec2(-alpha,0.0));

	if(fract(alpha) < .5){
	// if(mod(fract(alpha)*100.0f,repeat) < repeat*0.5){
	// if(pos.x < 0.5){
		// right eye (right half of SBS);
		// return vec4(0.0,0.0,1.0,1.0);
		return HOOKED_tex(vec2(0.5 + halfX, pos.y));
	}
	// left eye (left half of SBS)
	//return vec4(0.0);
	// return vec4(0.0,0.0,1.0,1.0);
	return HOOKED_tex(vec2(halfX, pos.y));
}

vec4 hook(){

	vec4 myColor = vec4(0.0,0.0,0.0,0.1);//HOOKED_tex(HOOKED_pos);

	float alpha = ( HOOKED_pos.x + HOOKED_pos.y * tilt ) * pitch_adjusted - center;

	// if(gl_FragCoord.x > HOOKED_size.x * 0.5){
	// 	return vec4(0.0,0.1,0.0,1.0);
	// }

	// This makes a perfect red/cyan filter somehow
	// float alpha = gl_FragCoord.x; // + gl_FragCoord.y;

    // we sample 3 times since the r,g,b subpixels for each "original" pixel needs to be additionally shifted by one extra "subpixel" amount per channel to match the unique sub-pixel layout of the LKGP display
    // myColor = my_sample(alpha);
    myColor.r = my_sample(alpha).r;
    myColor.g = my_sample(alpha + subp).g;
    myColor.b = my_sample(alpha + 2.0f * subp).b;

    return myColor;
}
