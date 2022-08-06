#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D currentFrameTex;
uniform sampler2D lastFrameTex;
uniform int frameCount;

#define MAX_FRAMES 60.0f

void main()
{
    vec3 col = texture(currentFrameTex, TexCoords).rgb;
    vec3 col2 = texture(lastFrameTex, TexCoords).rgb;

    col = mix(col, col2, min(frameCount/MAX_FRAMES,1.0f));
    FragColor = vec4(col, 1.0);
}