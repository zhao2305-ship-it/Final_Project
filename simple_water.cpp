// =====================================================
// simple_water.cpp (FULL INTEGRATED)
// Water + Caustics + Refraction + Cube + Walls + Pinball
// Ball is TRUE 3D SPHERE (replaces old disc ball).
//
// Assets: Hachimi.bmp in same folder
//
// Controls:
// - Mouse LMB drag: orbit camera
// - Mouse wheel: zoom
// - READY: Left/Right arrows adjust launch angle
// - Enter: launch (or restart after WIN/LOSE)
// - R: reset to READY
// - 1/2/3: Calm / Windy / Storm
// - ESC: quit
// =====================================================

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- Math helpers ----------------
struct Vec2 { float x,y; };
struct Vec3 { float x,y,z; };

static Vec2 v2(float x,float y){ return {x,y}; }
static Vec3 v3(float x,float y,float z){ return {x,y,z}; }

static float clampf(float x,float a,float b){ return std::max(a,std::min(b,x)); }

static Vec3 add(Vec3 a, Vec3 b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static Vec3 sub(Vec3 a, Vec3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static Vec3 mul(Vec3 a,float s){ return {a.x*s,a.y*s,a.z*s}; }
static float dot(Vec3 a,Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3 cross(Vec3 a, Vec3 b){
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
static Vec3 normalize(Vec3 a){
    float l = std::sqrt(dot(a,a));
    if(l < 1e-8f) return {0,1,0};
    return {a.x/l, a.y/l, a.z/l};
}

// 4x4 column-major matrices
static void matIdentity(float m[16]){
    for(int i=0;i<16;i++) m[i]=0;
    m[0]=m[5]=m[10]=m[15]=1;
}
static void matMul(const float a[16], const float b[16], float out[16]){
    float r[16];
    for(int row=0; row<4; row++){
        for(int col=0; col<4; col++){
            r[col*4 + row] = 0.0f;
            for(int k=0;k<4;k++){
                r[col*4 + row] += a[k*4 + row]*b[col*4 + k];
            }
        }
    }
    for(int i=0;i<16;i++) out[i]=r[i];
}
static void makePerspective(float fovyDeg, float aspect, float zNear, float zFar, float out[16]){
    float f = 1.0f / std::tan(fovyDeg * 0.5f * (float)M_PI / 180.0f);
    for (int i=0;i<16;i++) out[i]=0;
    out[0]  = f / aspect;
    out[5]  = f;
    out[10] = (zFar + zNear) / (zNear - zFar);
    out[11] = -1.0f;
    out[14] = (2.0f * zFar * zNear) / (zNear - zFar);
}
static void makeLookAt(Vec3 eye, Vec3 center, Vec3 up, float out[16]){
    Vec3 f = normalize(sub(center, eye));
    Vec3 s = normalize(cross(f, normalize(up)));
    Vec3 u = cross(s, f);

    float m[16]; matIdentity(m);
    m[0]=s.x; m[4]=s.y; m[8]=s.z;
    m[1]=u.x; m[5]=u.y; m[9]=u.z;
    m[2]=-f.x; m[6]=-f.y; m[10]=-f.z;

    m[12] = -dot(s, eye);
    m[13] = -dot(u, eye);
    m[14] =  dot(f, eye);

    for(int i=0;i<16;i++) out[i]=m[i];
}
static void makeTranslate(float tx,float ty,float tz,float out[16]){
    matIdentity(out);
    out[12]=tx; out[13]=ty; out[14]=tz;
}
static void makeScale(float sx,float sy,float sz,float out[16]){
    matIdentity(out);
    out[0]=sx; out[5]=sy; out[10]=sz;
}

// ---------------- Texture loading (BMP via SDL) ----------------
static GLuint loadBMPTexture(const char* path)
{
    SDL_Surface* surf = SDL_LoadBMP(path);
    if (!surf) {
        std::fprintf(stderr, "SDL_LoadBMP failed for '%s': %s\n", path, SDL_GetError());
        return 0;
    }

    SDL_Surface* rgba = SDL_ConvertSurfaceFormat(surf, SDL_PIXELFORMAT_ABGR8888, 0);
    SDL_FreeSurface(surf);
    if (!rgba) {
        std::fprintf(stderr, "SDL_ConvertSurfaceFormat failed: %s\n", SDL_GetError());
        return 0;
    }

    GLuint tex=0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rgba->w, rgba->h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgba->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);
    SDL_FreeSurface(rgba);
    return tex;
}

// ---------------- Shaders ----------------
// uMode:
// 0 = ground (underwater refraction + caustics + depth)
// 1 = water surface (sparkle specular, alpha)
// 2 = cube (underwater tint + caustics)
// 3 = flat unlit color (walls, markers, line)
// 5 = 3D ball (sphere lighting)
static const char* VS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec3 aNormal;

uniform mat4  uMVP;
uniform mat4  uModel;
uniform float uTime;
uniform int   uMode;

uniform float uAmpScale;
uniform vec2  uDir[3];
uniform float uAmp[3];
uniform float uFreq[3];
uniform float uSpeed[3];

// Ripples (impact)
uniform int   uRippleCount;
uniform vec2  uRipplePos[8];
uniform float uRippleT0[8];
uniform float uRippleAmp[8];

out vec2 vUV;
out vec3 vWorldPos;
out vec3 vWorldN;

float rippleHeight(vec2 xz, float t)
{
    float h = 0.0;
    for(int i=0;i<uRippleCount;i++){
        float age = t - uRippleT0[i];
        if(age < 0.0) continue;

        float d = distance(xz, uRipplePos[i]);

        float waveSpeed = 3.4;
        float k = 12.0;
        float decayT = 1.9;
        float decayD = 1.35;

        float phase = k * (d - waveSpeed * age);
        float env = exp(-decayT * age) * exp(-decayD * d);

        float nearBoost = 1.0 / (1.0 + 2.5*d);

        h += uRippleAmp[i] * sin(phase) * env * nearBoost;
    }
    return h;
}

void main()
{
    vec3 pos = aPos;

    if (uMode == 1) {
        vec2 xz0 = pos.xz;

        float dispY = 0.0;
        vec2  dispXZ = vec2(0.0);

        float sBase = clamp(0.55 * (uAmpScale / 3.0), 0.10, 0.55);

        for(int i=0;i<3;i++){
            vec2 D = normalize(uDir[i]);
            float A = uAmp[i] * uAmpScale;
            float w = uFreq[i];
            float phi = w * dot(D, xz0) + uSpeed[i] * uTime;

            float Q = sBase / (max(0.0001, w*A) * 3.0);

            dispY  += A * sin(phi);
            dispXZ += Q * A * cos(phi) * D;
        }

        dispY += rippleHeight(xz0, uTime);

        pos.x += dispXZ.x;
        pos.z += dispXZ.y;
        pos.y += dispY;
    }

    vec4 wp = uModel * vec4(pos, 1.0);
    vWorldPos = wp.xyz;

    vWorldN = normalize(mat3(uModel) * aNormal);
    vUV = aUV;

    gl_Position = uMVP * vec4(pos, 1.0);
}
)";

static const char* FS = R"(#version 330 core
in vec2 vUV;
in vec3 vWorldPos;
in vec3 vWorldN;

out vec4 FragColor;

uniform float uTime;
uniform int   uMode;

uniform vec3  uCamPos;
uniform vec3  uLightDir;
uniform vec3  uSkyColor;
uniform vec3  uWaterColor;

uniform float uAmpScale;

uniform sampler2D uGroundTex;

uniform vec2  uDir[3];
uniform float uAmp[3];
uniform float uFreq[3];
uniform float uSpeed[3];

// Ripples
uniform int   uRippleCount;
uniform vec2  uRipplePos[8];
uniform float uRippleT0[8];
uniform float uRippleAmp[8];

// Flat / ball color
uniform vec3  uFlatColor;

float saturate(float x){ return clamp(x, 0.0, 1.0); }

float rippleHeight(vec2 xz, float t)
{
    float h = 0.0;
    for(int i=0;i<uRippleCount;i++){
        float age = t - uRippleT0[i];
        if(age < 0.0) continue;

        float d = distance(xz, uRipplePos[i]);

        float waveSpeed = 3.4;
        float k = 12.0;
        float decayT = 1.9;
        float decayD = 1.35;

        float phase = k * (d - waveSpeed * age);
        float env = exp(-decayT * age) * exp(-decayD * d);

        float nearBoost = 1.0 / (1.0 + 2.5*d);

        h += uRippleAmp[i] * sin(phase) * env * nearBoost;
    }
    return h;
}

vec3 waterNormal(vec2 xz, float t)
{
    float dhdx = 0.0;
    float dhdz = 0.0;

    for (int i = 0; i < 3; ++i) {
        float phase = uFreq[i] * dot(uDir[i], xz) + uSpeed[i] * t;
        float c = cos(phase);
        float A = uAmp[i] * uAmpScale;

        dhdx += A * c * uFreq[i] * uDir[i].x;
        dhdz += A * c * uFreq[i] * uDir[i].y;
    }

    float eps = 0.02;
    float h0  = rippleHeight(xz, t);
    float hx  = rippleHeight(xz + vec2(eps,0.0), t);
    float hz  = rippleHeight(xz + vec2(0.0,eps), t);
    dhdx += (hx - h0) / eps;
    dhdz += (hz - h0) / eps;

    return normalize(vec3(-dhdx, 1.0, -dhdz));
}

vec3 addHighFreqNormal(vec3 N, vec2 xz, float t)
{
    float n1 = sin(xz.x * 38.0 + t * 4.6);
    float n2 = cos(xz.y * 46.0 - t * 4.1);
    float n3 = sin((xz.x + xz.y) * 30.0 + t * 3.1);
    float n4 = cos((xz.x - xz.y) * 34.0 - t * 3.6);

    vec3 hf = normalize(vec3(n1 + 0.6*n3, 3.3, n2 + 0.6*n4));
    float amp01 = saturate(uAmpScale/3.0);
    float k = mix(0.12, 0.22, amp01);
    return normalize(mix(N, hf, k));
}

float causticsFromWaves(vec2 xz, float t)
{
    float eps = 0.02;

    vec3 N0 = waterNormal(xz, t);
    vec3 Nx = waterNormal(xz + vec2(eps, 0.0), t);
    vec3 Nz = waterNormal(xz + vec2(0.0, eps), t);

    float kx = (Nx.x - N0.x) / eps;
    float kz = (Nz.z - N0.z) / eps;

    float curvature = abs(kx) + abs(kz);

    float amp01 = saturate(uAmpScale / 3.0);
    float strength = mix(0.75, 2.2, amp01);

    float f = curvature * strength;
    float c = pow(saturate(f * 0.9), 3.2);

    float jitter = 0.94 + 0.06 * sin(dot(xz, vec2(12.3, 7.7)) + t * 2.4);
    return saturate(c * jitter);
}

void main()
{
    float amp01 = saturate(uAmpScale / 3.0);

    // Flat unlit (walls/markers/line)
    if(uMode == 3){
        FragColor = vec4(uFlatColor, 1.0);
        return;
    }

    // 3D Ball (sphere lighting)
    if(uMode == 5){
        vec3 N = normalize(vWorldN);
        vec3 V = normalize(uCamPos - vWorldPos);

        float NdotL = max(dot(N, uLightDir), 0.0);
        vec3 diffuse = uFlatColor * (0.22 + 0.90 * NdotL);

        vec3 H = normalize(uLightDir + V);
        float spec = pow(max(dot(N, H), 0.0), 140.0) * 0.35;

        // subtle fresnel
        float cosTheta = saturate(dot(N, V));
        float fres = 0.04 + (1.0 - 0.04) * pow(1.0 - cosTheta, 5.0);

        vec3 color = diffuse + vec3(1.0)*spec;
        color = mix(color, uSkyColor, fres * 0.12);

        FragColor = vec4(color, 1.0);
        return;
    }

    // Water surface
    if (uMode == 1) {
        vec2 xz = vWorldPos.xz;

        vec3 N = waterNormal(xz, uTime);
        N = addHighFreqNormal(N, xz, uTime);

        vec3 V = normalize(uCamPos - vWorldPos);

        float NdotL = max(dot(N, uLightDir), 0.0);
        vec3 diffuse = vec3(0.05) + vec3(0.30) * NdotL;

        vec3 H = normalize(uLightDir + V);
        float nh = max(dot(N, H), 0.0);

        float spec1 = pow(nh, 420.0);
        float spec2 = pow(nh, 110.0) * 0.30;

        float sparkle = 0.62
            + (0.28 + 0.08*amp01) * sin(uTime * 6.5 + xz.x * 11.0)
            + (0.18 + 0.06*amp01) * sin(uTime * 9.2 + xz.y * 13.0);
        sparkle = clamp(sparkle, 0.25, 1.25);

        vec3 specCol = vec3(1.0) * (spec1 + spec2) * sparkle * (1.30 + 0.20*amp01);

        float cosTheta = saturate(dot(N, V));
        float F0 = 0.04;
        float fresnel = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 3.0);

        vec3 base = mix(uWaterColor, uSkyColor, fresnel);
        vec3 color = base * diffuse + specCol;
        color += uSkyColor * (fresnel * (0.20 + 0.08*amp01));

        FragColor = vec4(color, 0.55);
        return;
    }

    // Ground under water
    if (uMode == 0) {
        float depth = saturate(-vWorldPos.y * 0.45);

        vec2 xz = vWorldPos.xz;
        vec3 Nw = waterNormal(xz, uTime);

        vec3 I = vec3(0.0, -1.0, 0.0);
        float eta = 1.0 / 1.33;
        vec3 R = refract(I, normalize(Nw), eta);

        float refrStrength = (0.045 + 0.030 * amp01) * (1.0 - 0.55*depth);
        vec2 refrOffset = R.xz * refrStrength;

        vec3 img = texture(uGroundTex, vUV + refrOffset).rgb;

        float c = causticsFromWaves(xz, uTime);
        float shallowMask = pow(1.0 - depth, 0.65);
        c *= shallowMask;

        float causticStrength = 0.10 + 0.10 * amp01;
        img += vec3(1.0) * c * causticStrength;

        vec3 waterTint = vec3(0.80, 0.92, 1.03);
        img = mix(img, img * waterTint, 0.22 + 0.18*depth);
        img = mix(img, vec3(0.74, 0.87, 1.00), depth * 0.12);

        FragColor = vec4(img, 1.0);
        return;
    }

    // Underwater cube
    {
        float depth = saturate(-vWorldPos.y * 0.45);
        vec2 xz = vWorldPos.xz;

        vec3 N = normalize(vWorldN);
        vec3 V = normalize(uCamPos - vWorldPos);
        float NdotL = max(dot(N, uLightDir), 0.0);

        vec3 baseColor = vec3(0.85, 0.35, 0.25);
        vec3 diffuse = baseColor * (0.22 + 0.88 * NdotL);

        vec3 H = normalize(uLightDir + V);
        float spec = pow(max(dot(N, H), 0.0), 90.0) * 0.20;

        vec3 color = diffuse + vec3(1.0) * spec;

        float c = causticsFromWaves(xz, uTime);
        c *= pow(1.0 - depth, 0.65);
        float causticStrength = 0.05 + 0.05 * amp01;
        color += vec3(1.0) * c * causticStrength;

        vec3 fogCol = vec3(0.10, 0.35, 0.55);
        vec3 tint   = vec3(0.75, 0.92, 1.05);
        color *= tint;
        color = mix(color, fogCol, depth * 0.35);

        FragColor = vec4(color, 1.0);
        return;
    }
}
)";

// --------------- GL helpers ---------------
static GLuint compileShader(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        char log[4096];
        glGetShaderInfoLog(s, 4096, nullptr, log);
        std::fprintf(stderr, "Shader compile error:\n%s\n", log);
    }
    return s;
}
static GLuint makeProgram(const char* vs, const char* fs){
    GLuint v=compileShader(GL_VERTEX_SHADER, vs);
    GLuint f=compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p=glCreateProgram();
    glAttachShader(p,v);
    glAttachShader(p,f);
    glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){
        char log[4096];
        glGetProgramInfoLog(p, 4096, nullptr, log);
        std::fprintf(stderr, "Program link error:\n%s\n", log);
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// --------------- Wave presets ---------------
struct Preset {
    float amp[3];
    float freq[3];
    float speed[3];
    float dir[6]; // (x,z) * 3
};
static void normalize2(float& x, float& z){
    float l = std::sqrt(x*x + z*z);
    if(l < 1e-6f){ x=1.0f; z=0.0f; return; }
    x/=l; z/=l;
}
static Preset Calm(){
    Preset p{};
    p.amp[0]=0.028f; p.amp[1]=0.018f; p.amp[2]=0.012f;
    p.freq[0]=1.7f;  p.freq[1]=2.3f;  p.freq[2]=3.0f;
    p.speed[0]=1.1f; p.speed[1]=0.9f; p.speed[2]=0.7f;
    p.dir[0]=1.0f; p.dir[1]=0.2f;
    p.dir[2]=0.2f; p.dir[3]=1.0f;
    p.dir[4]=0.7f; p.dir[5]=0.7f;
    return p;
}
static Preset Windy(){
    Preset p{};
    p.amp[0]=0.060f; p.amp[1]=0.040f; p.amp[2]=0.028f;
    p.freq[0]=2.5f;  p.freq[1]=3.3f;  p.freq[2]=4.0f;
    p.speed[0]=1.9f; p.speed[1]=1.6f; p.speed[2]=1.3f;
    p.dir[0]=1.0f; p.dir[1]=0.0f;
    p.dir[2]=0.8f; p.dir[3]=0.6f;
    p.dir[4]=0.2f; p.dir[5]=1.0f;
    return p;
}
static Preset Storm(){
    Preset p{};
    p.amp[0]=0.105f; p.amp[1]=0.075f; p.amp[2]=0.055f;
    p.freq[0]=3.0f;  p.freq[1]=4.2f;  p.freq[2]=5.1f;
    p.speed[0]=2.8f; p.speed[1]=2.4f; p.speed[2]=2.0f;
    p.dir[0]=1.0f; p.dir[1]=0.3f;
    p.dir[2]=0.9f; p.dir[3]=0.5f;
    p.dir[4]=0.6f; p.dir[5]=0.9f;
    return p;
}
static void uploadPreset(GLuint prog, const Preset& p,
                         GLint locAmp, GLint locFreq, GLint locSpeed, GLint locDir)
{
    float dir[6] = { p.dir[0],p.dir[1], p.dir[2],p.dir[3], p.dir[4],p.dir[5] };
    for(int i=0;i<3;i++){
        float dx=dir[i*2+0], dz=dir[i*2+1];
        normalize2(dx,dz);
        dir[i*2+0]=dx; dir[i*2+1]=dz;
    }
    glUseProgram(prog);
    glUniform1fv(locAmp, 3, p.amp);
    glUniform1fv(locFreq, 3, p.freq);
    glUniform1fv(locSpeed, 3, p.speed);
    glUniform2fv(locDir, 3, dir);
}

// --------------- Mesh builders ---------------
// Vertex layout: pos(3), uv(2), normal(3) => 8 floats
static void buildGrid(int GRID, float size, std::vector<float>& verts, std::vector<unsigned int>& idx)
{
    verts.clear(); idx.clear();
    verts.reserve((GRID+1)*(GRID+1)*8);

    for(int z=0; z<=GRID; z++){
        for(int x=0; x<=GRID; x++){
            float fx = (float)x/(float)GRID - 0.5f;
            float fz = (float)z/(float)GRID - 0.5f;
            float u  = (float)x/(float)GRID;
            float v  = (float)z/(float)GRID;

            verts.push_back(fx*size);
            verts.push_back(0.0f);
            verts.push_back(fz*size);
            verts.push_back(u);
            verts.push_back(v);
            verts.push_back(0.0f);
            verts.push_back(1.0f);
            verts.push_back(0.0f);
        }
    }
    for(int z=0; z<GRID; z++){
        for(int x=0; x<GRID; x++){
            unsigned int i0 = z*(GRID+1)+x;
            unsigned int i1 = i0+1;
            unsigned int i2 = i0+(GRID+1);
            unsigned int i3 = i2+1;
            idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
            idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
        }
    }
}

static void buildCube(std::vector<float>& v, std::vector<unsigned int>& idx)
{
    struct V { float px,py,pz, u,v, nx,ny,nz; };
    std::vector<V> verts; verts.reserve(24);

    auto pushFace = [&](Vec3 n, Vec3 a, Vec3 b, Vec3 c, Vec3 d){
        verts.push_back({a.x,a.y,a.z, 0,0, n.x,n.y,n.z});
        verts.push_back({b.x,b.y,b.z, 1,0, n.x,n.y,n.z});
        verts.push_back({c.x,c.y,c.z, 1,1, n.x,n.y,n.z});
        verts.push_back({d.x,d.y,d.z, 0,1, n.x,n.y,n.z});
    };

    float s=0.5f;
    pushFace(v3(0,0,1),   v3(-s,-s, s), v3( s,-s, s), v3( s, s, s), v3(-s, s, s));
    pushFace(v3(0,0,-1),  v3( s,-s,-s), v3(-s,-s,-s), v3(-s, s,-s), v3( s, s,-s));
    pushFace(v3(1,0,0),   v3( s,-s, s), v3( s,-s,-s), v3( s, s,-s), v3( s, s, s));
    pushFace(v3(-1,0,0),  v3(-s,-s,-s), v3(-s,-s, s), v3(-s, s, s), v3(-s, s,-s));
    pushFace(v3(0,1,0),   v3(-s, s, s), v3( s, s, s), v3( s, s,-s), v3(-s, s,-s));
    pushFace(v3(0,-1,0),  v3(-s,-s,-s), v3( s,-s,-s), v3( s,-s, s), v3(-s,-s, s));

    idx.clear(); idx.reserve(36);
    for(int f=0; f<6; f++){
        unsigned int base = f*4;
        idx.push_back(base+0); idx.push_back(base+1); idx.push_back(base+2);
        idx.push_back(base+0); idx.push_back(base+2); idx.push_back(base+3);
    }

    v.clear(); v.reserve(verts.size()*8);
    for(auto& e: verts){
        v.push_back(e.px); v.push_back(e.py); v.push_back(e.pz);
        v.push_back(e.u);  v.push_back(e.v);
        v.push_back(e.nx); v.push_back(e.ny); v.push_back(e.nz);
    }
}

// Quad on XZ plane centered at origin, UV 0..1
static void buildQuadXZ(std::vector<float>& v, std::vector<unsigned int>& idx)
{
    v = {
        -0.5f, 0.0f, -0.5f,   0.0f,0.0f,   0,1,0,
         0.5f, 0.0f, -0.5f,   1.0f,0.0f,   0,1,0,
         0.5f, 0.0f,  0.5f,   1.0f,1.0f,   0,1,0,
        -0.5f, 0.0f,  0.5f,   0.0f,1.0f,   0,1,0,
    };
    idx = {0,1,2, 0,2,3};
}

// NEW: UV Sphere (pos, uv, normal) => 8 floats/vertex
static void buildSphere(int segU, int segV, std::vector<float>& v, std::vector<unsigned int>& idx)
{
    v.clear(); idx.clear();
    v.reserve((segU+1)*(segV+1)*8);

    for(int y=0; y<=segV; y++){
        float vv = (float)y/(float)segV;          // 0..1
        float phi = vv * (float)M_PI;             // 0..pi
        float sy = std::cos(phi);
        float sr = std::sin(phi);

        for(int x=0; x<=segU; x++){
            float uu = (float)x/(float)segU;      // 0..1
            float theta = uu * 2.0f*(float)M_PI;  // 0..2pi

            float sx = sr * std::cos(theta);
            float sz = sr * std::sin(theta);

            // position on unit sphere
            v.push_back(sx);
            v.push_back(sy);
            v.push_back(sz);

            // uv
            v.push_back(uu);
            v.push_back(vv);

            // normal (same as pos on unit sphere)
            v.push_back(sx);
            v.push_back(sy);
            v.push_back(sz);
        }
    }

    for(int y=0; y<segV; y++){
        for(int x=0; x<segU; x++){
            unsigned int i0 = y*(segU+1)+x;
            unsigned int i1 = i0+1;
            unsigned int i2 = i0+(segU+1);
            unsigned int i3 = i2+1;

            idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
            idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
        }
    }
}

// --------------- GL buffer setup ---------------
struct Mesh {
    GLuint vao=0,vbo=0,ebo=0;
    int indexCount=0;
};
static Mesh uploadMesh(const std::vector<float>& v, const std::vector<unsigned int>& idx)
{
    Mesh m{};
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, v.size()*sizeof(float), v.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(unsigned int), idx.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    m.indexCount = (int)idx.size();
    return m;
}
static void destroyMesh(Mesh& m){
    if(m.ebo) glDeleteBuffers(1,&m.ebo);
    if(m.vbo) glDeleteBuffers(1,&m.vbo);
    if(m.vao) glDeleteVertexArrays(1,&m.vao);
    m = {};
}

// --------------- Dynamic line (GL_LINES) ---------------
struct LineGPU {
    GLuint vao=0,vbo=0;
};
static LineGPU createLineGPU(){
    LineGPU L{};
    glGenVertexArrays(1,&L.vao);
    glGenBuffers(1,&L.vbo);
    glBindVertexArray(L.vao);
    glBindBuffer(GL_ARRAY_BUFFER,L.vbo);

    float data[16] = {
        0,0,0,  0,0,  0,1,0,
        0,0,0,  1,1,  0,1,0
    };
    glBufferData(GL_ARRAY_BUFFER,sizeof(data),data,GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(5*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    return L;
}
static void updateLineGPU(LineGPU& L, Vec3 a, Vec3 b){
    float data[16] = {
        a.x,a.y,a.z,  0,0,  0,1,0,
        b.x,b.y,b.z,  1,1,  0,1,0
    };
    glBindBuffer(GL_ARRAY_BUFFER,L.vbo);
    glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(data),data);
}
static void destroyLineGPU(LineGPU& L){
    if(L.vbo) glDeleteBuffers(1,&L.vbo);
    if(L.vao) glDeleteVertexArrays(1,&L.vao);
    L = {};
}

// ---------------- Ripple system ----------------
struct Ripple {
    Vec2 pos;     // xz
    float t0;
    float amp;
    bool active=false;
};

static void pushRipple(std::vector<Ripple>& ripples, Vec2 p, float t, float amp)
{
    int best = -1;
    float oldest = 1e9f;
    for(int i=0;i<(int)ripples.size();i++){
        if(!ripples[i].active){ best=i; break; }
        if(ripples[i].t0 < oldest){ oldest=ripples[i].t0; best=i; }
    }
    if(best < 0) return;
    ripples[best].active = true;
    ripples[best].pos = p;
    ripples[best].t0 = t;
    ripples[best].amp = amp;
}

static void packRipples(const std::vector<Ripple>& r,
                        int& count, float pos2[16], float t0[8], float amp[8], float nowT)
{
    count = 0;
    for(int i=0;i<(int)r.size() && count<8;i++){
        if(!r[i].active) continue;
        if(nowT - r[i].t0 > 3.0f) continue;
        pos2[count*2+0] = r[i].pos.x;
        pos2[count*2+1] = r[i].pos.y;
        t0[count] = r[i].t0;
        amp[count] = r[i].amp;
        count++;
    }
}

// ---------------- Game state ----------------
enum GameState { READY, RUNNING, WIN, LOSE };

// Ball vs AABB in XZ
static bool ballAABB(Vec2 p, float r, Vec2 bmin, Vec2 bmax, Vec2& outNormal, float& outPen)
{
    float cx = clampf(p.x, bmin.x, bmax.x);
    float cz = clampf(p.y, bmin.y, bmax.y);

    float dx = p.x - cx;
    float dz = p.y - cz;

    float d2 = dx*dx + dz*dz;
    if(d2 >= r*r) return false;

    float inside = (p.x >= bmin.x && p.x <= bmax.x && p.y >= bmin.y && p.y <= bmax.y) ? 1.0f : 0.0f;

    if(inside > 0.5f){
        float left   = std::abs(p.x - bmin.x);
        float right  = std::abs(bmax.x - p.x);
        float bottom = std::abs(p.y - bmin.y);
        float top    = std::abs(bmax.y - p.y);

        float m = std::min(std::min(left,right), std::min(bottom,top));
        if(m == left)   { outNormal = v2(-1,0); outPen = r + left; }
        else if(m==right){ outNormal = v2(1,0); outPen = r + right; }
        else if(m==bottom){ outNormal = v2(0,-1); outPen = r + bottom; }
        else { outNormal = v2(0,1); outPen = r + top; }
    }else{
        float d = std::sqrt(std::max(d2, 1e-8f));
        outNormal = v2(dx/d, dz/d);
        outPen = r - d;
    }
    return true;
}

int main(int, char**)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    int winW=960, winH=540;
    SDL_Window* window = SDL_CreateWindow(
        "Water + Pinball (Sphere Ball)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (!window) {
        std::fprintf(stderr, "CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GLContext ctx = SDL_GL_CreateContext(window);
    if (!ctx) {
        std::fprintf(stderr, "CreateContext failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "GLEW init failed\n");
        return 1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint prog = makeProgram(VS, FS);

    // Uniform locations
    GLint locMVP      = glGetUniformLocation(prog, "uMVP");
    GLint locModel    = glGetUniformLocation(prog, "uModel");
    GLint locTime     = glGetUniformLocation(prog, "uTime");
    GLint locMode     = glGetUniformLocation(prog, "uMode");
    GLint locCamPos   = glGetUniformLocation(prog, "uCamPos");

    GLint locAmpScale = glGetUniformLocation(prog, "uAmpScale");
    GLint locDir      = glGetUniformLocation(prog, "uDir");
    GLint locAmp      = glGetUniformLocation(prog, "uAmp");
    GLint locFreq     = glGetUniformLocation(prog, "uFreq");
    GLint locSpeed    = glGetUniformLocation(prog, "uSpeed");

    GLint locLightDir = glGetUniformLocation(prog, "uLightDir");
    GLint locSkyColor = glGetUniformLocation(prog, "uSkyColor");
    GLint locWaterCol = glGetUniformLocation(prog, "uWaterColor");

    GLint locGroundTex= glGetUniformLocation(prog, "uGroundTex");
    GLint locFlatColor= glGetUniformLocation(prog, "uFlatColor");

    // Ripples uniforms
    GLint locRippleCount = glGetUniformLocation(prog, "uRippleCount");
    GLint locRipplePos   = glGetUniformLocation(prog, "uRipplePos");
    GLint locRippleT0    = glGetUniformLocation(prog, "uRippleT0");
    GLint locRippleAmp   = glGetUniformLocation(prog, "uRippleAmp");

    GLuint groundTex = loadBMPTexture("Hachimi.bmp");
    if (groundTex == 0) {
        std::fprintf(stderr, "Put Hachimi.bmp next to the exe and try again.\n");
        return 1;
    }

    // Meshes
    std::vector<float> v; std::vector<unsigned int> idx;

    buildGrid(140, 5.0f, v, idx);
    Mesh grid = uploadMesh(v, idx);

    buildCube(v, idx);
    Mesh cube = uploadMesh(v, idx);

    buildQuadXZ(v, idx);
    Mesh quad = uploadMesh(v, idx);

    // NEW: sphere mesh for ball
    buildSphere(32, 20, v, idx);
    Mesh ballSphere = uploadMesh(v, idx);

    LineGPU line = createLineGPU();

    // Light & palette
    Vec3 lightDir = normalize(v3(-0.35f, 1.0f, 0.25f));
    Vec3 sky      = v3(0.70f, 0.85f, 1.00f);
    Vec3 waterCol = v3(0.02f, 0.35f, 0.55f);

    // Waves
    Preset preset = Windy();
    glUseProgram(prog);
    uploadPreset(prog, preset, locAmp, locFreq, locSpeed, locDir);

    float ampScale = 1.0f;

    // Board bounds
    const float half = 2.5f;
    const float wallT = 0.12f;
    const float wallH = 0.25f;
    const float waterY = 0.0f;
    const float groundY = -1.0f;

    // Entrance / goal
    Vec2 entrance = v2(-half + 0.25f, 0.0f);
    Vec2 goalC    = v2( half - 0.25f, 0.0f);
    Vec2 goalHalf = v2(0.18f, 0.35f);

    // Ball (XZ in Vec2)
    float ballR = 0.12f;
    const float ballSpeed = 2.8f; // constant
    Vec2 ballPos = entrance;
    Vec2 ballVel = v2(1,0);

    // Launch angle control
    float launchAngle = 0.0f; // radians
    const float angleStep = 1.6f * (float)M_PI / 180.0f;
    const float maxAngle = 70.0f * (float)M_PI / 180.0f;

    // Central cube collider in XZ (AABB)
    Vec2 cubeMin = v2(-0.5f, -0.5f);
    Vec2 cubeMax = v2( 0.5f,  0.5f);

    auto resetGame = [&](){
        ballPos = entrance;
        ballVel = v2(std::cos(launchAngle), std::sin(launchAngle));
    };

    // Timer & state
    GameState state = READY;
    float timeLimit = 15.0f;
    float timeLeft = timeLimit;

    // Ripples
    std::vector<Ripple> ripples(8);
    auto hitRipple = [&](Vec2 p, float intensity, float nowT){
        float a = clampf(0.10f + 0.12f * intensity, 0.14f, 0.36f);
        pushRipple(ripples, p, nowT, a);
    };

    // Orbit camera
    float yaw = 0.8f;
    float pitch = 0.55f;
    float radius= 7.5f;
    Vec3 target= v3(0.0f, 0.0f, 0.0f);

    bool lmb=false;
    int lastX=0, lastY=0;

    Uint64 start = SDL_GetPerformanceCounter();
    Uint64 perfFreq  = SDL_GetPerformanceFrequency();

    bool running=true;

    while(running){
        Uint64 now = SDL_GetPerformanceCounter();
        float t = float(now - start) / float(perfFreq);

        static Uint64 prev = now;
        float dt = float(now - prev) / float(perfFreq);
        prev = now;
        dt = std::clamp(dt, 0.0f, 0.033f);

        SDL_Event e;
        while(SDL_PollEvent(&e)){
            if(e.type==SDL_QUIT) running=false;

            if(e.type==SDL_WINDOWEVENT && e.window.event==SDL_WINDOWEVENT_SIZE_CHANGED){
                winW = e.window.data1;
                winH = e.window.data2;
            }

            if(e.type==SDL_MOUSEBUTTONDOWN){
                if(e.button.button==SDL_BUTTON_LEFT){ lmb=true; lastX=e.button.x; lastY=e.button.y; }
            }
            if(e.type==SDL_MOUSEBUTTONUP){
                if(e.button.button==SDL_BUTTON_LEFT) lmb=false;
            }
            if(e.type==SDL_MOUSEWHEEL){
                radius *= (e.wheel.y > 0) ? 0.90f : 1.10f;
                radius = std::clamp(radius, 3.0f, 25.0f);
            }
            if(e.type==SDL_MOUSEMOTION){
                int mx=e.motion.x, my=e.motion.y;
                int dx=e.motion.x - lastX;
                int dy=e.motion.y - lastY;
                lastX=mx; lastY=my;

                if(lmb){
                    yaw   += dx * 0.008f;
                    pitch += dy * 0.008f;
                    pitch = std::clamp(pitch, -1.2f, 1.2f);
                }
            }

            if(e.type==SDL_KEYDOWN){
                SDL_Keycode k = e.key.keysym.sym;
                if(k==SDLK_ESCAPE) running=false;

                if(k==SDLK_1){ preset=Calm();  uploadPreset(prog,preset,locAmp,locFreq,locSpeed,locDir); }
                if(k==SDLK_2){ preset=Windy(); uploadPreset(prog,preset,locAmp,locFreq,locSpeed,locDir); }
                if(k==SDLK_3){ preset=Storm(); uploadPreset(prog,preset,locAmp,locFreq,locSpeed,locDir); }

                if(k==SDLK_r){
                    state = READY;
                    timeLeft = timeLimit;
                    resetGame();
                }

                if(state==READY){
                    if(k==SDLK_LEFT){
                        launchAngle -= angleStep;
                        launchAngle = std::clamp(launchAngle, -maxAngle, maxAngle);
                        resetGame();
                    }
                    if(k==SDLK_RIGHT){
                        launchAngle += angleStep;
                        launchAngle = std::clamp(launchAngle, -maxAngle, maxAngle);
                        resetGame();
                    }
                }

                if(k==SDLK_RETURN || k==SDLK_KP_ENTER){
                    if(state==READY || state==WIN || state==LOSE){
                        state = RUNNING;
                        timeLeft = timeLimit;
                        resetGame();
                        hitRipple(ballPos, 2.0f, t);
                    }
                }
            }
        }

        // Update game
        if(state == RUNNING){
            timeLeft -= dt;
            if(timeLeft <= 0.0f){
                timeLeft = 0.0f;
                state = LOSE;
            }

            // enforce constant speed
            float len = std::sqrt(ballVel.x*ballVel.x + ballVel.y*ballVel.y);
            if(len < 1e-6f){ ballVel = v2(1,0); len=1.0f; }
            ballVel.x = (ballVel.x/len) * ballSpeed;
            ballVel.y = (ballVel.y/len) * ballSpeed;

            // integrate
            ballPos.x += ballVel.x * dt;
            ballPos.y += ballVel.y * dt;

            // wall collision (reflect + ripple)
            bool hit=false;

            if(ballPos.x - ballR < -half){
                ballPos.x = -half + ballR;
                ballVel.x = std::abs(ballVel.x);
                hit=true;
            }
            if(ballPos.x + ballR > half){
                ballPos.x = half - ballR;
                ballVel.x = -std::abs(ballVel.x);
                hit=true;
            }
            if(ballPos.y - ballR < -half){
                ballPos.y = -half + ballR;
                ballVel.y = std::abs(ballVel.y);
                hit=true;
            }
            if(ballPos.y + ballR > half){
                ballPos.y = half - ballR;
                ballVel.y = -std::abs(ballVel.y);
                hit=true;
            }
            if(hit){
                hitRipple(ballPos, 3.0f, t);
            }

            // cube collision (ball vs AABB)
            Vec2 n; float pen=0;
            if(ballAABB(ballPos, ballR, cubeMin, cubeMax, n, pen)){
                // push out
                ballPos.x += n.x * pen;
                ballPos.y += n.y * pen;

                // reflect velocity v' = v - 2*(v·n)*n
                float vd = ballVel.x*n.x + ballVel.y*n.y;
                ballVel.x = ballVel.x - 2.0f*vd*n.x;
                ballVel.y = ballVel.y - 2.0f*vd*n.y;

                hitRipple(ballPos, 4.0f, t);
            }

            // goal check
            if(std::abs(ballPos.x - goalC.x) <= goalHalf.x &&
               std::abs(ballPos.y - goalC.y) <= goalHalf.y)
            {
                state = WIN;
                hitRipple(ballPos, 4.5f, t);
            }
        }

        // HUD title
        {
            std::string st;
            if(state==READY){
                float deg = launchAngle * 180.0f / (float)M_PI;
                st = "READY (Left/Right aim, Enter launch)  Angle=" + std::to_string((int)deg) + " deg";
            } else if(state==RUNNING){
                st = "RUNNING  TimeLeft: " + std::to_string((int)std::ceil(timeLeft)) + "s";
            } else if(state==WIN){
                st = "WIN! (Press Enter to play again, R to reset)";
            } else {
                st = "LOSE! (Press Enter to retry, R to reset)";
            }
            SDL_SetWindowTitle(window, st.c_str());
        }

        // Camera position
        Vec3 camPos = add(target, v3(
            radius * std::cos(pitch) * std::sin(yaw),
            radius * std::sin(pitch),
            radius * std::cos(pitch) * std::cos(yaw)
        ));

        // Matrices
        float proj[16], view[16], pv[16];
        makePerspective(45.0f, (float)winW/(float)winH, 0.1f, 60.0f, proj);
        makeLookAt(camPos, target, v3(0,1,0), view);
        matMul(proj, view, pv);

        // Pack ripples
        int rCount=0;
        float rPos[16]{};
        float rT0[8]{};
        float rAmp[8]{};
        packRipples(ripples, rCount, rPos, rT0, rAmp, t);

        // Update aim line
        Vec3 lineA = v3(ballPos.x, waterY + 0.08f, ballPos.y);
        Vec3 lineB = v3(ballPos.x + std::cos(launchAngle)*1.2f, waterY + 0.08f, ballPos.y + std::sin(launchAngle)*1.2f);
        updateLineGPU(line, lineA, lineB);

        // Render
        glViewport(0,0,winW,winH);
        glClearColor(0.92f,0.96f,1.00f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);

        // common uniforms
        glUniform1f(locTime, t);
        glUniform3f(locCamPos, camPos.x, camPos.y, camPos.z);
        glUniform3f(locLightDir, lightDir.x, lightDir.y, lightDir.z);
        glUniform3f(locSkyColor, sky.x, sky.y, sky.z);
        glUniform3f(locWaterCol, waterCol.x, waterCol.y, waterCol.z);
        glUniform1f(locAmpScale, ampScale);

        // ripple uniforms
        glUniform1i(locRippleCount, rCount);
        glUniform2fv(locRipplePos, rCount, rPos);
        glUniform1fv(locRippleT0, rCount, rT0);
        glUniform1fv(locRippleAmp, rCount, rAmp);

        // texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, groundTex);
        glUniform1i(locGroundTex, 0);

        // Draw ground
        {
            float M[16]; makeTranslate(0.0f, groundY, 0.0f, M);
            float MVPm[16]; matMul(pv, M, MVPm);

            glUniformMatrix4fv(locModel, 1, GL_FALSE, M);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 0);

            glBindVertexArray(grid.vao);
            glDrawElements(GL_TRIANGLES, grid.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // Goal zone marker
        {
            float Tm[16], Sm[16], M[16];
            makeTranslate(goalC.x, groundY+0.01f, goalC.y, Tm);
            makeScale(goalHalf.x*2.0f, 1.0f, goalHalf.y*2.0f, Sm);
            matMul(Tm, Sm, M);

            float MVPm[16]; matMul(pv, M, MVPm);
            glUniformMatrix4fv(locModel, 1, GL_FALSE, M);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 3);
            glUniform3f(locFlatColor, 0.20f, 0.85f, 0.35f);

            glBindVertexArray(quad.vao);
            glDrawElements(GL_TRIANGLES, quad.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // Entrance marker
        {
            float Tm[16], Sm[16], M[16];
            makeTranslate(entrance.x, groundY+0.01f, entrance.y, Tm);
            makeScale(0.22f, 1.0f, 0.22f, Sm);
            matMul(Tm, Sm, M);

            float MVPm[16]; matMul(pv, M, MVPm);
            glUniformMatrix4fv(locModel, 1, GL_FALSE, M);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 3);
            glUniform3f(locFlatColor, 0.95f, 0.55f, 0.20f);

            glBindVertexArray(quad.vao);
            glDrawElements(GL_TRIANGLES, quad.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // Walls (flat)
        auto drawWall = [&](float cx,float cz,float sx,float sz){
            float Tm[16], Sm[16], M[16];
            makeTranslate(cx, waterY + wallH*0.5f, cz, Tm);
            makeScale(sx, wallH, sz, Sm);
            matMul(Tm, Sm, M);

            float MVPm[16]; matMul(pv, M, MVPm);
            glUniformMatrix4fv(locModel, 1, GL_FALSE, M);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 3);
            glUniform3f(locFlatColor, 0.20f, 0.22f, 0.25f);

            glBindVertexArray(quad.vao);
            glDrawElements(GL_TRIANGLES, quad.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        };
        drawWall(-half + wallT*0.5f, 0.0f, wallT, half*2.0f);
        drawWall( half - wallT*0.5f, 0.0f, wallT, half*2.0f);
        drawWall(0.0f, -half + wallT*0.5f, half*2.0f, wallT);
        drawWall(0.0f,  half - wallT*0.5f, half*2.0f, wallT);

        // Central cube (visual)
        {
            float Tm[16], Sm[16], M[16];
            makeTranslate(0.0f, -0.55f, 0.0f, Tm);
            makeScale(1.0f, 1.0f, 1.0f, Sm);
            matMul(Tm, Sm, M);

            float MVPm[16]; matMul(pv, M, MVPm);
            glUniformMatrix4fv(locModel, 1, GL_FALSE, M);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 2);

            glBindVertexArray(cube.vao);
            glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // NEW: Draw 3D ball sphere
        {
            float Tm[16], Sm[16], M[16];
            // center the ball slightly above water so it reads clearly as 3D
            makeTranslate(ballPos.x, waterY + ballR, ballPos.y, Tm);
            makeScale(ballR, ballR, ballR, Sm);
            matMul(Tm, Sm, M);

            float MVPm[16]; matMul(pv, M, MVPm);
            glUniformMatrix4fv(locModel, 1, GL_FALSE, M);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 5);

            if(state==WIN) glUniform3f(locFlatColor, 0.25f, 0.95f, 0.45f);
            else if(state==LOSE) glUniform3f(locFlatColor, 0.95f, 0.25f, 0.25f);
            else glUniform3f(locFlatColor, 0.96f, 0.96f, 0.98f);

            glBindVertexArray(ballSphere.vao);
            glDrawElements(GL_TRIANGLES, ballSphere.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // Aim line (READY only)
        if(state==READY){
            float I[16]; matIdentity(I);
            float MVPm[16]; matMul(pv, I, MVPm);
            glUniformMatrix4fv(locModel, 1, GL_FALSE, I);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 3);
            glUniform3f(locFlatColor, 1.0f, 0.95f, 0.35f);

            glBindVertexArray(line.vao);
            glLineWidth(3.0f);
            glDrawArrays(GL_LINES, 0, 2);
            glBindVertexArray(0);
        }

        // Water surface (last, transparent)
        {
            float I[16]; matIdentity(I);
            float MVPm[16]; matMul(pv, I, MVPm);

            glUniformMatrix4fv(locModel, 1, GL_FALSE, I);
            glUniformMatrix4fv(locMVP,   1, GL_FALSE, MVPm);
            glUniform1i(locMode, 1);

            glBindVertexArray(grid.vao);
            glDrawElements(GL_TRIANGLES, grid.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        SDL_GL_SwapWindow(window);
    }

    // cleanup
    glDeleteTextures(1, &groundTex);
    glDeleteProgram(prog);

    destroyMesh(grid);
    destroyMesh(cube);
    destroyMesh(quad);
    destroyMesh(ballSphere);
    destroyLineGPU(line);

    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
