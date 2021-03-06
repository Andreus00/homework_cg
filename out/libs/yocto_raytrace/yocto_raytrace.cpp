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
static vec4f shade_raytrace(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params);

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

// volumetric function

vec3f volumetric_recursion(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng, const raytrace_params& params,
    const instance_data& instance, const material_data& material) {
  if (bounce > params.bounces) {
    return vec3f{0, 0, 0};
  }
  auto out_isec = intersect_bvh(bvh, scene, ray);
  // se non c'?? intersezione ritorno l'environment
  if (!out_isec.hit) return eval_environment(scene, ray.d);
  // prendo le informazioni sull'oggetto che ?? stato colpito per capire se
  // il raggio sta uscendo dall'oggetto volumetric o no
  auto& out_instance = scene.instances[out_isec.instance];
  auto& out_material = scene.materials[out_instance.material];
  auto& out_shape    = scene.shapes[out_instance.shape];
  // position
  auto out_position = transform_point(
      instance.frame, eval_position(out_shape, out_isec.element, out_isec.uv));
  // normal
  auto out_normal = transform_direction(
      instance.frame, eval_normal(out_shape, out_isec.element, out_isec.uv));
  auto dist         = distance(out_position, ray.o);
  auto hit_distance = (-1 / material.ior) * log(rand1f(rng));
  // if (rand1f(rng) > 0.01) std::cout << dist << " " << hit_distance <<
  // std::endl;
  if (hit_distance > dist) {  // se non hitta faccio passare oltre il raggio
    return material.color * rgba_to_rgb(shade_raytrace(scene, bvh,
                                ray3f{ray.o, ray.d}, bounce, rng, params));
  } else {
    auto x = eval_transmittance(
        vec3f{material.ior, material.ior, material.ior}, hit_distance);
    auto rand_dir     = sample_sphere(rand2f(rng));
    auto bounce_ratio = hit_distance / dist;
    auto bounce_point = bounce_ratio * ray.o +
                        (1 - bounce_ratio) * out_position;
    return x + volumetric_recursion(scene, bvh, ray3f{bounce_point, rand_dir},
                   bounce + 1, rng, params, instance, material);
  }
}

////////////////////////
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
    auto rg        = eval_texcoord(shape, isec.element, isec.uv);
    auto textcoord = vec2f{fmod(rg.x, 1), fmod(rg.y, 1)};

    ///////////////////////////////////////////////////
    //
    // compute material values according to textures
    //

    // color

    vec3f color = material.color;

    auto color_texture = eval_texture(
        scene, material.color_tex, textcoord, true);

    color *= rgba_to_rgb(color_texture);

    // normal
    color *= rgba_to_rgb(
        eval_texture(scene, material.normal_tex, textcoord, true));

    // emission
    color *= rgba_to_rgb(
        eval_texture(scene, material.emission_tex, textcoord, true));

    // roughness
    color *= rgba_to_rgb(
        eval_texture(scene, material.roughness_tex, textcoord, true));

    // scattering
    color *= rgba_to_rgb(
        eval_texture(scene, material.scattering_tex, textcoord, true));

    //////////////////////////
    //
    // handle opacity
    //
    auto color_shp = eval_color(scene, instance, isec.element, isec.uv);
    if (rand1f(rng) < (1 - material.opacity * color_texture.w * color_shp.w)) {
      return shade_raytrace(
          scene, bvh, ray3f{position, ray.d}, bounce + 1, rng, params);
    }

    //////////////////////////
    //
    // recursion
    // matte, glossy, reflective, transparent, refractive, subsurface,
    // volumetric, gltfpbr

    auto outgoing = -ray.d;

    auto radiance = material.emission;
    if (bounce >= params.bounces) return rgb_to_rgba(radiance);

    // handle points, lines and triangles

    if (!shape.points.empty()) {
      normal = -ray.d;
    } else if (!shape.lines.empty()) {
      normal = orthonormalize(-ray.d, normal);
    } else if (!shape.triangles.empty()) {
      if (dot(-ray.d, normal) < 0) normal = -normal;
    }

    switch (material.type) {
      case material_type::matte: {  // diffuse
        auto incoming = sample_hemisphere_cos(normal, rand2f(rng));
        radiance += color *
                    rgba_to_rgb(shade_raytrace(scene, bvh,
                        ray3f{position, incoming}, bounce + 1, rng, params));
        break;
      }
      case (material_type::reflective): {
        if ((material.roughness <= 0)) {  // polished metals
          auto incoming = reflect(outgoing, normal);
          radiance += fresnel_schlick(color, normal, outgoing) *
                      rgba_to_rgb(shade_raytrace(scene, bvh,
                          ray3f{position, incoming}, bounce + 1, rng, params));
        } else {  // rough metals
          auto exponent = 2 / pow((float)material.roughness, (int)4);
          auto halfway  = sample_hemisphere_cospower(
               exponent, normal, rand2f(rng));
          auto incoming = reflect(outgoing, halfway);
          radiance += fresnel_schlick(color, halfway, outgoing) *
                      rgba_to_rgb(shade_raytrace(scene, bvh,
                          ray3f{position, incoming}, bounce + 1, rng, params));
        }
        break;
      }
      case material_type::glossy: {  // rough plastic
        auto exponent = 2 / pow((float)material.roughness, (int)4);
        auto halfway  = sample_hemisphere_cospower(
             exponent, normal, rand2f(rng));
        if (rand1f(rng) <
            fresnel_schlick({0.04, 0.04, 0.04}, halfway, outgoing).x) {
          auto incoming = reflect(outgoing, halfway);
          radiance += rgba_to_rgb(shade_raytrace(
              scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params));
        } else {
          auto incoming = sample_hemisphere_cos(normal, rand2f(rng));
          radiance += color *
                      rgba_to_rgb(shade_raytrace(scene, bvh,
                          ray3f{position, incoming}, bounce + 1, rng, params));
        }
        break;
      }
      case material_type::transparent: {  // polished dielectrics
        auto test = fresnel_schlick({0.04, 0.04, 0.04}, normal, -outgoing);
        if (rand1f(rng) < test.x) {
          auto incoming = reflect(outgoing, normal);
          radiance += rgba_to_rgb(shade_raytrace(
              scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params));
        } else {
          radiance += color *
                      rgba_to_rgb(shade_raytrace(scene, bvh,
                          ray3f{position, -outgoing}, bounce + 1, rng, params));
        }
        break;
      }
      case material_type::refractive: {  // refraction
        auto test = fresnel_schlick({0.04, 0.04, 0.04}, normal, -outgoing);
        if (rand1f(rng) < test.x) {
          auto incoming = reflect(outgoing, normal);
          radiance += rgba_to_rgb(shade_raytrace(
              scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params));
        } else {
          auto ior = material.ior;

          if (dot(ray.d, normal) > 0) {
            normal = -normal;
          } else {
            ior = 1 / ior;
          }

          radiance += color *
                      rgba_to_rgb(shade_raytrace(scene, bvh,
                          ray3f{position, refract(outgoing, normal, ior)},
                          bounce + 1, rng, params));
        }
        break;
      }
      case material_type::volumetric: {
        // vedo vesro dove va il raggio
        radiance += volumetric_recursion(scene, bvh, ray3f{position, ray.d},
            bounce, rng, params, instance, material);
        break;
      }
      default: {
        std::cout << material_type_names[(int)material.type] << std::endl;
      }
    }
    return rgb_to_rgba(radiance);
  }
  return rgb_to_rgba(eval_environment(scene, ray.d));
}

static vec4f shade_toon(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  auto isec = intersect_bvh(bvh, scene, ray);
  if (isec.hit) {
    auto& scene_instance = scene.instances[isec.instance];
    auto& scene_shape    = scene.shapes[scene_instance.shape];
    auto  normal         = transform_direction(
                 scene_instance.frame, eval_normal(scene_shape, isec.element, isec.uv));
    // position
    auto position = transform_point(scene_instance.frame,
        eval_position(scene_shape, isec.element, isec.uv));

    auto& material = scene.materials[scene_instance.material];
    if (material.type == material_type::reflective) {
      auto incoming = reflect(-ray.d, normal);
      return rgb_to_rgba(fresnel_schlick(material.color, normal, -ray.d)) *
             shade_toon(
                 scene, bvh, ray3f{position, incoming}, bounce, rng, params);
    }

    // texture coordinates
    auto  rg           = eval_texcoord(scene_shape, isec.element, isec.uv);
    auto  textcoord    = vec2f{fmod(rg.x, 1), fmod(rg.y, 1)};
    auto  color        = material.color * rgba_to_rgb(eval_texture(scene,
                                              material.color_tex, textcoord, true));
    vec3f illumination = zero3f;
    // contatore del numero di luci che colpiscono il punto
    int counter = 0;
    // itero sulle luci e vedo se il dot product tra la normale del punto
    // colpito e il punto in cui is trova la luce.
    auto test_rim         = 0;
    int  instance_counter = -1;
    for (auto& el : scene.instances) {
      instance_counter++;
      auto emission = scene.materials.at(el.material).emission;
      if (emission.x != 0 || emission.y != 0 || emission.z != 0) {
        // posizione della luce
        auto light_world_position = (el.frame.x + el.frame.y + el.frame.z) *
                                    el.frame.o;

        auto check_isec = intersect_bvh(bvh, scene,
            ray3f{position, normalize(light_world_position - position)});

        float hit_distance = 0.0f;

        if (check_isec.hit) {
          auto& check_instance = scene.instances[check_isec.instance];
          // position
          auto check_position = (check_instance.frame.x +
                                    check_instance.frame.y +
                                    check_instance.frame.z) *
                                check_instance.frame.o;
          hit_distance = distance(check_position, position);
        }

        if (hit_distance >= distance(light_world_position, position) - 0.02) {
          auto pos = normalize(light_world_position);
          // vettore che dal punto va verso la luce
          auto light_vector = normalize(pos - position);

          // check per vedere se ci sono ogetti tra la luce e il punto
          // todo
          float cosin           = dot(light_vector, normal);
          auto  light_intensity = (1 - 1 / emission);
          if (cosin > 0) {
            illumination += light_intensity;
            counter++;
          }
        } else {
          color *= 0.1;
        }
      }
    }
    if (scene.instance_names.at(isec.instance) != "floor") {
      float rim_dot       = 1 - dot(-ray.d, normal);
      float rim_intensity = smoothstep(0.7f - 0.01f, 0.7f + 0.01f, rim_dot);

      illumination += rim_intensity;
    }

    // add the light reflection
    auto outgoing = reflect(-ray.d, normal);

    auto isec = intersect_bvh(bvh, scene, ray3f{position, outgoing});
    if (isec.hit) {
      auto& ins = scene.instances[isec.instance];
      auto& mat = scene.materials.at(ins.material);
      if (mat.emission.x != 0 || mat.emission.y != 0 || mat.emission.z != 0) {
        illumination += (1 - 1 / mat.emission);
      }
    }
    // if the point is not hitted by a light, I reduce the signal of the colors

    // add the color of the environment
    auto env_emission = vec3f{1, 1, 1};
    for (auto& env : scene.environments) {
      illumination += env.emission;
    }
    return rgb_to_rgba(color * illumination);
  }
  return rgb_to_rgba(eval_environment(scene, ray.d));
}

static vec4f easy_shade_toon(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  auto isec = intersect_bvh(bvh, scene, ray);
  if (isec.hit) {
    auto& scene_instance = scene.instances[isec.instance];
    auto& scene_shape    = scene.shapes[scene_instance.shape];
    auto  normal         = transform_direction(
                 scene_instance.frame, eval_normal(scene_shape, isec.element, isec.uv));
    auto& material = scene.materials[scene_instance.material];

    auto  a     = dot(normal, ray.o);
    auto& shape = scene.shapes[scene_instance.shape];

    auto rg        = eval_texcoord(shape, isec.element, isec.uv);
    auto textcoord = vec2f{fmod(rg.x, 1), fmod(rg.y, 1)};

    vec3f color = material.color;

    auto color_texture = eval_texture(
        scene, material.color_tex, textcoord, true);

    color *= rgba_to_rgb(color_texture);
    if (a > 0.3) {
      return rgb_to_rgba(color);
    } else {
      return rgb_to_rgba(color * 0.1);
    }
  }
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
    case raytrace_shader_type::toon: return shade_toon;
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
