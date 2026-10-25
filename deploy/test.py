import acl
om = "/home/HwHiAiUser/snap/lightstereo_onnx/weights/BANet-KITTI.om"  # 或绝对路径

acl.init()
acl.rt.set_device(0)
model_id, ret = acl.mdl.load_from_file(om)
assert ret == 0, ret
desc = acl.mdl.create_desc()
assert acl.mdl.get_desc(desc, model_id) == 0

for i in range(acl.mdl.get_num_inputs(desc)):
    print("IN ", i, acl.mdl.get_input_name_by_index(desc, i))
for i in range(acl.mdl.get_num_outputs(desc)):
    print("OUT", i, acl.mdl.get_output_name_by_index(desc, i))

acl.mdl.destroy_desc(desc)
acl.mdl.unload(model_id)
acl.rt.reset_device(0)
acl.finalize()