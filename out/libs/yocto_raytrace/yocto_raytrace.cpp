//
// Implementation for Yocto/RayTrace.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "yocto_raytrace.h"

#include <yocto/yocto_cli.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_shading.h>
#include <yocto/yocto_shape.h>

#include <iostream>

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR SCENE EVALUATION
// -----------------------------------------------------------------------------
namespace yocto {

// Generates a ray from a camera for yimg::image plane coordinate uv and
// the lens coordinates luv.
static ray3f eval_camera(const camera_data& camera, const vec2f& uv) {
  auto film = camera.aspect >= 1
                  ? vec2f{camera.film, camera.film / camera.aspect}
                  : vec2f{camera.film * camera.aspect, camera.film};
  auto q    = transform_point(camera.frame,
         {film.x * (0.5f - uv.x), film.y * (uv.y - 0.5f), camera.lens});
  auto e    = transform_point(camera.frame, {0, 0, 0});
  return {e, normalize(e - q)};
}

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------

// Raytrace renderer.
static vec4f shade_raytrace(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  auto isec = intersect_bvh(bvh, scene, ray);
  if (isec.hit) {
    auto& instance = scene.instances[isec.instance];
    auto& material = scene.materials[instance.material];
    auto& shape    = scene.shapes[instance.shape];
    ///////////////////////////////////
    //
    // compute position, normal and texcoord
    //

    // position
    auto position = transform_point(
        instance.frame, eval_position(shape, isec.element, isec.uv));

    // normal
    auto normal = transform_direction(
        instance.frame, eval_normal(shape, isec.element, isec.uv));

    // texture coordinates
    auto rgb       = eval_texcoord(shape, isec.element, isec.uv);
    auto textcoord = vec2f{fmod(rgb.x, 1), fmod(rgb.y, 1)};

    ///////////////////////////////////////////////////
    //
    // compute material values according to textures
    //

    // color
    vec3f color_texture = material.color;
    if (material.color_tex >= 0)
      color_texture *= rgba_to_rgb(
          eval_texture(scene.textures[material.color_tex], textcoord, true));

    // normal
    if (material.normal_tex >= 0)
      color_texture *= rgba_to_rgb(
          eval_texture(scene.textures[material.normal_tex], textcoord, true));

    // emission
    if (material.emission_tex >= 0)
      color_texture *= rgba_to_rgb(
          eval_texture(scene.textures[material.emission_tex], textcoord, true));

    // roughness
    if (material.roughness_tex >= 0)
      color_texture *= rgba_to_rgb(eval_texture(
          scene.textures[material.roughness_tex], textcoord, true));

    // scattering
    if (material.scattering_tex >= 0)
      color_texture *= rgba_to_rgb(eval_texture(
          scene.textures[material.roughness_tex], textcoord, true));

    // environment ?
    // auto environment_texture = eval_environment(scene, ray);

    //////////////////////////
    //
    // handle opacity
    //

    //////////////////////////
    //
    // recursion
    // matte, glossy, reflective, transparent, refractive, subsurface,
    // volumetric, gltfpbr

    auto radiance = material.emission;
    if (bounce >= params.bounces) return rgb_to_rgba(radiance);
    switch (material.type) {
      case material_type::matte: {  // diffuse
        auto incoming = sample_hemisphere_cos(normal, rand2f(rng));
        radiance += color_texture *
                    rgba_to_rgb(shade_raytrace(scene, bvh,
                        ray3f{position, incoming}, bounce + 1, rng, params));
        break;
      }
      case (material_type::reflective): {
        if ((material.roughness <= 0)) {  // polished metals
          auto incoming = reflect(ray.o, normal);
          radiance += fresnel_schlick(color_texture, normal, ray.o) *
                      rgba_to_rgb(shade_raytrace(scene, bvh,
                          ray3f{position, incoming}, bounce + 1, rng, params));
        } else {  // rough metals
          auto exponent = 2 / pow((float)material.roughness, (int)4);
          auto halfway  = sample_hemisphere_cospower(
               exponent, normal, rand2f(rng));
          auto incoming = reflect(ray.o, halfway);
          radiance += fresnel_schlick(color_texture, halfway, ray.o) *
                      rgba_to_rgb(shade_raytrace(scene, bvh,
                          ray3f{position, incoming}, bounce + 1, rng, params));
        }
        break;
      }
      case material_type::glossy: {  // rough plastic
      }
      case material_type::transparent: {  // polished dielectrics
        auto ks = 0.04f;
      }
    }
    return rgb_to_rgba(radiance);
  }
  return rgb_to_rgba(eval_environment(scene, ray.d));
}

// Matte renderer.
static vec4f shade_matte(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  return {0, 0, 0, 0};
}

// Eyelight renderer.
static vec4f shade_eyelight(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  auto isec = intersect_bvh(bvh, scene, ray);
  if (isec.hit) {
    auto& instance = scene.instances[isec.instance];
    auto& material = scene.materials[instance.material];
    auto& shape    = scene.shapes[instance.shape];
    auto  normal   = transform_direction(
           instance.frame, eval_normal(shape, isec.element, isec.uv));
    return rgb_to_rgba(material.color * dot(normal, -ray.d));
  }
  return {0, 0, 0, 0};
}

static vec4f shade_normal(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  auto isec = intersect_bvh(bvh, scene, ray);
  if (isec.hit) {
    auto& scene_instance = scene.instances[isec.instance];
    auto& scene_shape    = scene.shapes[scene_instance.shape];
    auto  normal         = transform_direction(
                 scene_instance.frame, eval_normal(scene_shape, isec.element, isec.uv));

    return rgb_to_rgba(0.5 + (normal * 0.5));
  }

  return {0, 0, 0, 0};
}

static vec4f shade_texcoord(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  auto isec = intersect_bvh(bvh, scene, ray);
  if (isec.hit) {
    auto& scene_instance = scene.instances[isec.instance];
    auto& scene_shape    = scene.shapes[scene_instance.shape];
    auto  rgb            = eval_texcoord(scene_shape, isec.element, isec.uv);
    return {fmod(rgb.x, 1), fmod(rgb.y, 1), 0, 0};
  }
  return {0, 0, 0, 0};
}

static vec4f shade_color(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  auto isec = intersect_bvh(bvh, scene, ray);
  if (isec.hit) {
    return rgb_to_rgba(scene.materials[isec.instance].color);
  }
  return zero4f;
}

// Trace a single ray from the camera using the given algorithm.
using raytrace_shader_func = vec4f (*)(const scene_data& scene,
    const bvh_scene& bvh, const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params);
static raytrace_shader_func get_shader(const raytrace_params& params) {
  switch (params.shader) {
    case raytrace_shader_type::raytrace: return shade_raytrace;
    case raytrace_shader_type::matte: return shade_matte;
    case raytrace_shader_type::eyelight: return shade_eyelight;
    case raytrace_shader_type::normal: return shade_normal;
    case raytrace_shader_type::texcoord: return shade_texcoord;
    case raytrace_shader_type::color: return shade_color;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}

// Build the bvh acceleration structure.
bvh_scene make_bvh(const scene_data& scene, const raytrace_params& params) {
  return make_bvh(scene, false, false, params.noparallel);
}

// Init a sequence of random number generators.
raytrace_state make_state(
    const scene_data& scene, const raytrace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = raytrace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image.assign(state.width * state.height, {0, 0, 0, 0});
  state.hits.assign(state.width * state.height, 0);
  state.rngs.assign(state.width * state.height, {});
  auto rng_ = make_rng(1301081);
  for (auto& rng : state.rngs) {
    rng = make_rng(961748941ull, rand1i(rng_, 1 << 31) / 2 + 1);
  }
  return state;
}

// Progressively compute an image by calling trace_samples multiple times.
void raytrace_samples(raytrace_state& state, const scene_data& scene,
    const bvh_scene& bvh, const raytrace_params& params) {
  if (state.samples >= params.samples) return;
  auto& camera = scene.cameras[params.camera];
  auto  shader = get_shader(params);
  state.samples += 1;
  if (params.samples == 1) {
    for (auto idx = 0; idx < state.width * state.height; idx++) {
      auto i = idx % state.width, j = idx / state.width;
      auto u = (i + 0.5f) / state.width, v = (j + 0.5f) / state.height;
      auto ray      = eval_camera(camera, {u, v});
      auto radiance = shader(scene, bvh, ray, 0, state.rngs[idx], params);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    }
  } else if (params.noparallel) {
    for (auto idx = 0; idx < state.width * state.height; idx++) {
      auto i = idx % state.width, j = idx / state.width;
      auto u        = (i + rand1f(state.rngs[idx])) / state.width,
           v        = (j + rand1f(state.rngs[idx])) / state.height;
      auto ray      = eval_camera(camera, {u, v});
      auto radiance = shader(scene, bvh, ray, 0, state.rngs[idx], params);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    }
  } else {
    parallel_for(state.width * state.height, [&](int idx) {
      auto i = idx % state.width, j = idx / state.width;
      auto u        = (i + rand1f(state.rngs[idx])) / state.width,
           v        = (j + rand1f(state.rngs[idx])) / state.height;
      auto ray      = eval_camera(camera, {u, v});
      auto radiance = shader(scene, bvh, ray, 0, state.rngs[idx], params);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    });
  }
}

// Check image type
static void check_image(
    const color_image& image, int width, int height, bool linear) {
  if (image.width != width || image.height != height)
    throw std::invalid_argument{"image should have the same size"};
  if (image.linear != linear)
    throw std::invalid_argument{
        linear ? "expected linear image" : "expected srgb image"};
}

// Get resulting render
color_image get_render(const raytrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_render(image, state);
  return image;
}
void get_render(color_image& image, const raytrace_state& state) {
  check_image(image, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = state.image[idx] * scale;
  }
}
}  // namespace yocto