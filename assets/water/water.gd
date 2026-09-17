@tool
extends MeshInstance3D
## Handles updating the displacement/normal maps for the water material as well as
## managing wave generation pipelines.

const WATER_MAT := preload('res://assets/water/mat_water.tres')
const SPRAY_MAT := preload('res://assets/water/mat_spray.tres')
const WATER_MESH_HIGH8K := preload('res://assets/water/clipmap_high_8k.obj')
const WATER_MESH_HIGH := preload('res://assets/water/clipmap_high.obj')
const WATER_MESH_LOW := preload('res://assets/water/clipmap_low.obj')

enum MeshQuality { LOW, HIGH, HIGH8K }

@export_group('Wave Parameters')
@export_color_no_alpha var water_color : Color = Color(0.1, 0.15, 0.18) :
	set(value): water_color = value; RenderingServer.global_shader_parameter_set(&'water_color', water_color.srgb_to_linear())

@export_color_no_alpha var foam_color : Color = Color(0.73, 0.67, 0.62) :
	set(value): foam_color = value; RenderingServer.global_shader_parameter_set(&'foam_color', foam_color.srgb_to_linear())

## The parameters for wave cascades. Each parameter set represents one cascade.
## Recreates all compute piplines whenever a cascade is added or removed!
@export var parameters : Array[WaveCascadeParameters] :
	set(value):
		var new_size := len(value)
		# All below logic is basically just required for using in the editor!
		for i in range(new_size):
			# Ensure all values in the array have an associated cascade
			if not value[i]: value[i] = WaveCascadeParameters.new()
			if not value[i].is_connected(&'scale_changed', _update_scales_uniform):
				value[i].scale_changed.connect(_update_scales_uniform)
			value[i].spectrum_seed = Vector2i(rng.randi_range(-10000, 10000), rng.randi_range(-10000, 10000))
			value[i].time = 120.0 + PI*i # We make sure to choose a time offset such that cascades don't interfere!
		parameters = value
		_setup_wave_generator()
		_update_scales_uniform()
		_setup_cpu_displacement_textures()

@export_group('Performance Parameters')

@export_enum('128x128:128', '256x256:256', '512x512:512', '1024x1024:1024') var map_size := 1024 :
	set(value):
		map_size = value
		_setup_wave_generator()

@export var mesh_quality := MeshQuality.HIGH :
	set(value):
		mesh_quality = value
		if mesh_quality == MeshQuality.LOW:
			mesh = WATER_MESH_LOW
		if mesh_quality == MeshQuality.HIGH:
			mesh = WATER_MESH_HIGH
		if mesh_quality == MeshQuality.HIGH8K:
			mesh = WATER_MESH_HIGH8K

## How many times the wave simulation should update per second.
## Note: This doesn't reduce the frame stutter caused by FFT calculation, only
##       minimizes GPU time taken by it!

@export_range(0, 60) var updates_per_second := 50.0 :
	set(value):
		next_update_time = next_update_time - (1.0/(updates_per_second + 1e-10) - 1.0/(value + 1e-10))
		updates_per_second = value

var wave_generator : WaveGenerator :
	set(value):
		if wave_generator: wave_generator.queue_free()
		wave_generator = value
		add_child(wave_generator)
var rng = RandomNumberGenerator.new()
var time := 0.0
var next_update_time := 0.0

var displacement_maps := Texture2DArrayRD.new()
var normal_maps := Texture2DArrayRD.new()

func _init() -> void:
	rng.set_seed(1234) # This seed gives big waves!

func _ready() -> void:
	RenderingServer.global_shader_parameter_set(&'water_color', water_color.srgb_to_linear())
	RenderingServer.global_shader_parameter_set(&'foam_color', foam_color.srgb_to_linear())

var update_textures:bool = true

var just_calculated_water:bool = false
func _process(delta : float) -> void:
	# Update waves once every 1.0/updates_per_second.
	just_calculated_water = false
	if updates_per_second == 0 or time >= next_update_time:
		var target_update_delta := 1.0 / (updates_per_second + 1e-10)
		var update_delta := delta if updates_per_second == 0 else target_update_delta + (time - next_update_time)
		next_update_time = time + target_update_delta
		_update_water(update_delta)
		#_copy_cpu_displacement_textures_buffer()
		if update_textures:
			_manage_cpu_displacement_textures_updates(delta)
		just_calculated_water = true
	time += delta

func _setup_wave_generator() -> void:
	if parameters.size() <= 0: return
	for param in parameters:
		param.should_generate_spectrum = true

	wave_generator = WaveGenerator.new()
	wave_generator.map_size = map_size
	wave_generator.init_gpu(maxi(2, parameters.size())) # FIXME: This is needed because my RenderContext API sucks...

	displacement_maps.texture_rd_rid = RID()
	normal_maps.texture_rd_rid = RID()
	displacement_maps.texture_rd_rid = wave_generator.descriptors[&'displacement_map'].rid
	normal_maps.texture_rd_rid = wave_generator.descriptors[&'normal_map'].rid

	RenderingServer.global_shader_parameter_set(&'num_cascades', parameters.size())
	RenderingServer.global_shader_parameter_set(&'displacements', displacement_maps)
	RenderingServer.global_shader_parameter_set(&'normals', normal_maps)

func _update_scales_uniform() -> void:
	var map_scales : PackedVector4Array; map_scales.resize(len(parameters))
	for i in len(parameters):
		var params := parameters[i]
		var uv_scale := Vector2.ONE / params.tile_length
		map_scales[i] = Vector4(uv_scale.x, uv_scale.y, params.displacement_scale, params.normal_scale)
	# No global shader parameter for arrays :(
	WATER_MAT.set_shader_parameter(&'map_scales', map_scales)
	SPRAY_MAT.set_shader_parameter(&'map_scales', map_scales)

func _update_water(delta : float) -> void:
	if wave_generator == null: _setup_wave_generator()
	wave_generator.update(delta, parameters)

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		displacement_maps.texture_rd_rid = RID()
		normal_maps.texture_rd_rid = RID()

# =============================================================================
#  displacement textures loading from gpu
# =============================================================================

var mutex: Mutex
var thread: Thread

#var _cpu_displacement_textures_buffer : Dictionary = {} # dict idx:img
var _cpu_displacement_textures : Dictionary = {} # dict idx:img
var _displacement_textures_total_update_interval:float = 1.0 / 120.0
var _displacement_textures_update_time:float = 0.0
var _actually_used_textures_idx:Array = [] # length == as there are cascades with height displacement > 0

var _texture_loading_index:int = 0
func _manage_cpu_displacement_textures_updates(delta) -> void:
	if len(_cpu_displacement_textures) < 1:
		return
	var time_per_texture:float = _displacement_textures_total_update_interval / float(len(_cpu_displacement_textures))
	var _cpu_displacement_textures_indeces = _cpu_displacement_textures.keys()
	_cpu_displacement_textures_indeces.sort()
	if _displacement_textures_update_time > time_per_texture:
		_texture_loading_index += 1
		if _texture_loading_index >= len(_cpu_displacement_textures):
			_texture_loading_index = 0
		_img_async_image_idx = _cpu_displacement_textures_indeces[_texture_loading_index]
		_update_cpu_displacement_textures()
		_cpu_displacement_textures[_img_async_image_idx] = _img_async_buffer
		_displacement_textures_update_time = 0.0
	_displacement_textures_update_time += delta

# func _copy_cpu_displacement_textures_buffer():
# 	if _cpu_displacement_textures_buffer and len(_cpu_displacement_textures_buffer) > 0:
# 		mutex.lock()
# 		for k in _cpu_displacement_textures_buffer.keys():
# 			_cpu_displacement_textures[k] = _cpu_displacement_textures_buffer[k]
# 		mutex.unlock()

# func _async_gpu_readback():
# 	for cascade_index in _cpu_displacement_textures.keys():
# 		_update_cpu_displacement_textures(cascade_index)

func _setup_cpu_displacement_textures() -> void:
	# load first_n_cascades_for_collision_detection textures
	var _actually_used_textures_idx:Array = []
	for i in range(len(parameters)):
		var cascade = parameters[i]
		if cascade.displacement_scale > 0.001:
			_actually_used_textures_idx.append(i)
	
	var rid_displacement_map = wave_generator.descriptors[&'displacement_map'].rid
	var device:RenderingDevice = RenderingServer.get_rendering_device()
	for i in range(len(parameters)):
		if i in _actually_used_textures_idx:
			var tex = device.texture_get_data(rid_displacement_map, i) # layer is the texture of the cascade with the same index
			var img:Image = Image.create_from_data(wave_generator.map_size, wave_generator.map_size, false, Image.FORMAT_RGBAH, tex)
			#_cpu_displacement_textures_buffer[i] = img
			_cpu_displacement_textures[i] = img
	mutex = Mutex.new()

var _img_async_buffer : Image
var _img_async_image_idx:int = 0
var _readback_pending: bool = false
func _update_cpu_displacement_textures() -> void:
	# If we have a already pending request
	if _readback_pending:
		return

	var rid_displacement_map = wave_generator.descriptors[&"displacement_map"].rid
	var device: RenderingDevice = RenderingServer.get_rendering_device()

	_readback_pending = true

	device.texture_get_data_async(
		rid_displacement_map,
		_img_async_image_idx,
		_on_displacement_texture_data
	)


func _on_displacement_texture_data(data: PackedByteArray) -> void:
	# The request is over
	_readback_pending = false

	# Convert the data received by the GPU in an image
	var img := Image.create_from_data(
		wave_generator.map_size,
		wave_generator.map_size,
		false,
		Image.FORMAT_RGBAH,
		data
	)
	_img_async_buffer = img

func _world_to_uv(W:Vector2, tile_length:Vector2) -> Vector2:
	return Vector2(
		(W[0] - tile_length.x * floor(W[0] / tile_length.x)) / tile_length.x,
		(W[1] - tile_length.y * floor(W[1] / tile_length.y)) / tile_length.y)

func get_height(world_pos:Vector3, steps:int=3) -> float:
	# needs to be translated to gdscript.
	# each cascade is accessed in parameters array
	# try to implement it just for the first cascade for now
	var world_pos_xz = Vector2(world_pos.x,world_pos.z)
	var cam_distance:float = (get_viewport().get_camera_3d().global_position - world_pos).length()
	var summed_height:float = 0.0
	for cascade_index in _cpu_displacement_textures.keys():
		var displacement_scale:float = parameters[cascade_index].displacement_scale
		var tile_length:Vector2 = parameters[cascade_index].tile_length
		var d:Vector2 = Vector2.ZERO
		var x:Vector2 = world_pos_xz
		var y:Vector2 = Vector2.ZERO
		var y_raw:Color=Color.BLACK
		# iteratively approximate the correct uv to get the height of the vertex that was displaced in the XZ-axis
		for i in range(steps):
			var img_v = _world_to_uv(x, tile_length) * map_size
			
			# If this function is called before the execution of
			# _update_cpu_displacement_textures
			if _cpu_displacement_textures[cascade_index] == null:
				return 0.0
			
			y_raw = _cpu_displacement_textures[cascade_index].get_pixelv(img_v)
			y = Vector2(y_raw.r,y_raw.b)
			x = world_pos_xz-y
		summed_height += y_raw.g * displacement_scale
	return summed_height
