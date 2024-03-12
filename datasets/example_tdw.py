'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-03-06 06:56:54
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-03-10 11:22:14
 # @ Description: This file is distributed under the MIT license.
'''

from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path
import json
from tqdm import tqdm
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import _Librarian, ModelLibrarian, MaterialLibrarian, ModelRecord, MaterialRecord
from tdw.librarian import SceneLibrarian
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera

from tdw.output_data import Images
import random
import numpy as np


class SceneRenderer(Controller):
    def __init__(self, library = "models_full", port=1071):
        self.rng: np.random.RandomState = np.random.RandomState(32)
        self._lib = self._get_librarian(library)
        self.scene_lib = SceneLibrarian()
        output_dir = self._get_output_directory()
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        self.output_dir = str(output_dir.resolve())
        Path(self.output_dir).joinpath("records.json").write_text(
            json.dumps({"records": self._get_visualizer_metadata()},))

        super().__init__(port, launch_build=False)

        commands = [self.get_add_scene(scene_name="empty_scene"),
                    {"$type": "simulate_physics",
                     "value": False},
                    {"$type": "set_screen_size",
                     "height": 1024,
                     "width": 1024},
                    {"$type": "set_render_quality",
                     "render_quality": 5},
                    {"$type": "set_shadow_strength",
                     "strength": 1.0},
                    {"$type": "set_img_pass_encoding",
                     "value": False}]
        commands.extend(TDWUtils.create_avatar(position=self._get_avatar_position()))
        commands.extend([{"$type": "set_pass_masks",
                          "pass_masks": ["_img"]},
                         {"$type": "send_images",
                          "frequency": "always"}])
        commands.extend(self._get_post_processing_commands())
        self.communicate(commands)

    def _get_output_directory(self) -> Path:
        return Path.home().joinpath("TDWImages/ModelImages")

    def _get_avatar_position(self) -> Dict[str, float]:
        return {"x": 1.57, "y": 2.5, "z": 3.56}

    def _get_post_processing_commands(self) -> List[dict]:
        return [{"$type": "set_screen_space_reflections",
                 "enabled": True},
                {"$type": "set_vignette",
                 "enabled": False},
                {"$type": "set_focus_distance",
                 "focus_distance": 8.0}]

    def _get_librarian(self, library: str) -> ModelLibrarian:
        return ModelLibrarian(library + ".json")

    def _get_visualizer_metadata(self) -> List[dict]:
        self._lib: ModelLibrarian
        records: List[dict] = []
        for record in self._lib.records:
            if record.do_not_use:
                continue
            records.append({"name": record.name,
                            "wnid": record.wnid,
                            "wcategory": record.wcategory})
        return records

    def _record_is_ok(self, record) -> bool:
        return not record.do_not_use


    def _get_output_directory(self) -> Path:
        return Path.cwd().joinpath("datasets/ExampleImages")

    def get_record(self, name: str):
        records = self._lib.records
        for i,record in enumerate(records):
            if record.name == name:break
        return record
    
    def run(self, objs, id = 0):

        for obj in objs:
            image = image = self.get_image(self.get_record(obj[0]), obj[1], obj[2])

        #TDWUtils.save_images(image, str(id), output_directory=self.output_dir, append_pass=False)

        #camera = ThirdPersonCamera(avatar_id="a", position={"x": 0, "y": 20, "z": 0}, look_at={"x": 0, "y": 0, "z": 0})
        #capture = ImageCapture(avatar_ids=["a"], pass_masks=["_img"], path=self.output_dir)

        TDWUtils.save_images(image, str(id), output_directory=self.output_dir, append_pass=False)

        

    def get_image(self, record: ModelRecord, s: float = 1., pos = None) -> Images:
        o_id = Controller.get_unique_id()
        scale = 1.6
        if pos is None: pos = (random.random() * scale, random.random() * 0, random.random() * scale)
        #s = TDWUtils.get_unit_scale(record) * 2
        # Add the model.
        # Scale the model and get an image.
        # Look at the model's centroid.
        print(pos)
        resp = self.communicate([{"$type": "add_object",
                                  "name": record.name,
                                  "url": record.get_url(),
                                  'position' : {"x": pos[0], "y": pos[1], "z": pos[2]},
                                  "scale_factor": record.scale_factor,
                                  "rotation": record.canonical_rotation,
                                  "id": o_id},
                                 {"$type": "scale_object",
                                  "id": o_id,
                                  "scale_factor": {"x": s, "y": s, "z": s}},
                                 {"$type": "look_at",
                                  "avatar_id": "a",
                                  "object_id": o_id,
                                  "use_centroid": True}
                                  
                                  ])
        # Destroy the model and unload the asset bundle.
        return Images(resp[0])

    def create_scene(self) -> List[dict]:
        width: int = self.rng.randint(12, 18)
        length: int = self.rng.randint(14, 20)
        room_arr: np.array = np.zeros(shape=(width, length), dtype=int)
        # Define the uppermost width-wise wall.
        turn_south_at = int(length * 0.75) + self.rng.randint(1, 3)
        for i in range(turn_south_at + 1):
            room_arr[0, i] = 1
        turn_west_at = int(width * 0.75) + self.rng.randint(0, 2)
        for i in range(turn_west_at + 1):
            room_arr[i, turn_south_at] = 1
        turn_north_at = turn_south_at - self.rng.randint(4, 6)
        for i in range(turn_north_at, turn_south_at):
            room_arr[turn_west_at, i] = 1
        turn_west_at_2 = self.rng.randint(4, 6)
        for i in range(turn_west_at_2, turn_west_at):
            room_arr[i, turn_north_at] = 1
        for i in range(turn_north_at):
            room_arr[turn_west_at_2, i] = 1
        for i in range(turn_west_at_2):
            room_arr[i, 0] = 1
        # Create interior walls.
        if self.rng.random() < 0.5:
            interior_wall_0 = range(turn_north_at + 1, turn_south_at - 1)
            interior_wall_1 = range(1, turn_west_at_2 - 1)
        else:
            interior_wall_0 = range(turn_north_at + 2, turn_south_at)
            interior_wall_1 = range(2, turn_west_at_2)
        for i in interior_wall_0:
            room_arr[turn_west_at_2, i] = 2
        for i in interior_wall_1:
            room_arr[i, turn_north_at] = 2
        # Convert the array to commands.
        exterior_walls: List[dict] = list()
        interior_walls: List[dict] = list()
        for ix, iy in np.ndindex(room_arr.shape):
            if room_arr[ix, iy] == 1:
                exterior_walls.append({"x": ix, "y": iy})
            elif room_arr[ix, iy] == 2:
                interior_walls.append({"x": ix, "y": iy})
        # load_scene typically gets sent by calling c.start()
        return [{"$type": "load_scene",
                 "scene_name": "ProcGenScene"},
                {"$type": "create_exterior_walls",
                 "walls": exterior_walls},
                {"$type": "create_interior_walls",
                 "walls": interior_walls}]

    def set_floor(self) -> List[dict]:
        materials = ["parquet_wood_mahogany", "parquet_long_horizontal_clean", "parquet_wood_red_cedar"]
        material_name = materials[self.rng.randint(0, len(materials))]
        texture_scale: float = float(self.rng.uniform(4, 4.5))
        return [self.get_add_material(material_name=material_name),
                {"$type": "set_floor_material",
                 "name": material_name},
                {"$type": "set_floor_texture_scale",
                 "scale": {"x": texture_scale, "y": texture_scale}},
                {"$type": "set_floor_color",
                 "color": {"r": float(self.rng.uniform(0.7, 1)),
                           "g": float(self.rng.uniform(0.7, 1)),
                           "b": float(self.rng.uniform(0.7, 1)),
                           "a": 1.0}}]

    def set_walls(self) -> List[dict]:
        materials = ["cinderblock_wall", "concrete_tiles_linear_grey", "old_limestone_wall_reinforced"]
        material_name = materials[self.rng.randint(0, len(materials))]
        texture_scale: float = float(self.rng.uniform(0.2, 0.3))
        return [self.get_add_material(material_name=material_name),
                {"$type": "set_proc_gen_walls_material",
                 "name": material_name},
                {"$type": "set_proc_gen_walls_texture_scale",
                 "scale": {"x": texture_scale, "y": texture_scale}},
                {"$type": "set_proc_gen_walls_color",
                 "color": {"r": float(self.rng.uniform(0.7, 1)),
                           "g": float(self.rng.uniform(0.7, 1)),
                           "b": float(self.rng.uniform(0.7, 1)),
                           "a": 1.0}}]

    def set_ceiling(self) -> List[dict]:
        materials = ["bricks_red_regular", "bricks_chatham_gray_used", "bricks_salem_matt_used"]
        material_name = materials[self.rng.randint(0, len(materials))]
        texture_scale: float = float(self.rng.uniform(0.1, 0.2))
        return [{"$type": "create_proc_gen_ceiling"},
                self.get_add_material(material_name=material_name),
                {"$type": "set_proc_gen_ceiling_material",
                 "name": material_name},
                {"$type": "set_proc_gen_ceiling_texture_scale",
                 "scale": {"x": texture_scale, "y": texture_scale}},
                {"$type": "set_proc_gen_ceiling_color",
                 "color": {"r": float(self.rng.uniform(0.7, 1)),
                           "g": float(self.rng.uniform(0.7, 1)),
                           "b": float(self.rng.uniform(0.7, 1)),
                           "a": 1.0}}]

if __name__ == "__main__":
    renderer = SceneRenderer()

    """
    objs = [
        ("lg_table_marble_green", 1.5, (1.5024342400341737, 0.0, 0.9095410556213936)),
       ("apple", 1., (1.5024342400341737, .6, 1.5095410556213936)),
    ]
    renderer.run(objs, 2)

            ("cabinet_36_two_door_wood_oak_white_composite", 1.0, (-1.5024342400341737, .0, .095410556213936)),
        ("round_bowl_small_beech", .7, (.3, 0.1, 1.3)),
        ("cabinet_36_wall_wood_beech_honey_composite", 1., (.3, 0.4, 0.8)),
        ("b04_orange_00", 1., (.0, .9, .0))
    """

    objs = [
            ("cabinet_36_two_door_wood_oak_white_composite", 1.0, (-1.5024342400341737, .0, .095410556213936)),
        ("round_bowl_small_beech", .7, (.3, 0.1, 1.3)),
        ("cabinet_36_wall_wood_beech_honey_composite", 1., (.3, 0.4, 0.8)),
        ("b04_orange_00", 1., (.0, .9, .0))
    ]
    renderer.run(objs, 6)
    