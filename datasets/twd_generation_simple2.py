from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images
from pathlib import Path


c = Controller()
object_id = c.get_unique_id()
object_id2 = c.get_unique_id()
object_id3 = c.get_unique_id()
object_id4 = c.get_unique_id()
object_id5 = c.get_unique_id()
from tdw.librarian import ModelLibrarian, MaterialLibrarian
material_record = MaterialLibrarian("materials_low.json").get_record("parquet_long_horizontal_clean")

material_name = "parquet_long_horizontal_clean"

model_record = "white_lounger_chair"
model_record2 = "wood_chair"

#model_record = "round_coaster_cherry"
full_library = ModelLibrarian("models_core.json")
model_record = full_library.get_record("sink_cabinet_unit_wood_beech_honey_chrome_composite")
#model_record2 = "sink_cabinet_unit_wood_beech_honey_chrome_composite"
model_record2 = full_library.get_record("molteni_turner_sofa")
model_record3 = full_library.get_record("b04_bowl_smooth")
model_record4 = full_library.get_record("toaster_002")
model_record5 = full_library.get_record("b04_banana")

dx = 0.0
dz = 0.2

commands = [TDWUtils.create_empty_room(12, 12),
            #TDWUtils.create_room_from_image("/Users/melkor/Downloads/1.jpg"),
            c.get_add_object(model_name = model_record.name,
                            position = {'x':0.5 + dx, 'y':-0., 'z':+0.4 + dz},
                            #library = "models_flex.json",
                            object_id = object_id),

            c.get_add_object(model_name = model_record2.name,
                             position = {"x": -0.7, "y": 0, "z": -0.3},
                             #library = "models_flex.json",
                             object_id = object_id2),

            c.get_add_object(model_name = model_record3.name,
                            position = {'x':0.3 + dx, 'y':1.0, 'z':+0.6 + dz},
                            #library = "models_flex.json",
                            object_id = object_id3),

            c.get_add_object(model_name = model_record4.name,
                            position = {'x':1.1 + dx, 'y':1.0, 'z':+0.4 + dz},
                            #library = "models_flex.json",
                            object_id = object_id4),

            c.get_add_object(model_name = model_record5.name,
                            position = {'x':1.1 + dx, 'y':1.0, 'z':+0.6 + dz},
                            #library = "models_flex.json",
                            object_id = object_id5),
            c.get_add_material(material_name=material_name),

                             ]

"""
            c.get_add_object(model_name = "cabinet_36_wall_wood_beech_honey_composite",
                            position = {"x": -0.7, 'y': .0, 'z':0.6},
                            object_id = c.get_unique_id()),
"""

commands.extend([]
    #TDWUtils.set_visual_material(c=c, substructure=model_record.substructure, material="wood_red_cedar", object_id=object_id),
)
commands.extend(
    #TDWUtils.set_visual_material(c=c, substructure=model_record.substructure, material="stone_surface_rough_cracks", object_id=object_id),
    TDWUtils.set_visual_material(c=c, substructure=model_record2.substructure, material="travertine_grey", object_id=object_id2),
)

commands.extend([
    #{"$type": "set_color", "color": {"r": 0.719607845, "g": 0.0156862754, "b": 0.1901961, "a": 1.0}, "id": object_id},
    #{"$type": "set_color", "color": {"r": 0.019607845, "g": 0.0156862754, "b": 0.7901961, "a": 1.0}, "id": object_id2},
    ]
)
                             
commands.extend(TDWUtils.create_avatar(position={"x": 1.7, "y": 1.6, "z": 1.6},
                                       avatar_id="a",
                                       look_at={"x": -0.0, "y": -0.0, "z": 0.2}))


commands.extend([{"$type": "set_pass_masks",
                  "pass_masks": ["_img", "_id"],
                  "avatar_id": "a"},
                 {"$type": "send_images",
                  "frequency": "always",
                  "ids": ["a"]}])

resp = c.communicate(commands)
output_directory = str(Path.cwd().joinpath("datasets/ExampleImages"))
print(f"Images will be saved to: {output_directory}")

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    # Get Images output data.
    if r_id == "imag":
        images = Images(resp[i])
        # Determine which avatar captured the image.
        if images.get_avatar_id() == "a":
            TDWUtils.save_images(images=images, filename="2", output_directory=output_directory)
c.communicate({"$type": "terminate"})