#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : The StanfordTown external environment to interate with the web interface
#           refs to `generative_agents maze.py`

import math
from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, Field, model_validator

from metagpt.environment.base_env import ExtEnv, mark_as_readable, mark_as_writeable
from metagpt.environment.stanford_town.env_space import (
    EnvAction,
    EnvActionType,
    EnvObsParams,
    EnvObsType,
    EnvObsValType,
    get_action_space,
    get_observation_space,
)
from metagpt.utils.common import read_csv_to_list, read_json_file


class StanfordTownExtEnv(ExtEnv):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    maze_asset_path: Optional[Path] = Field(default=None, description="迷宫资产的存储路径")
    maze_width: int = Field(default=140, description="迷宫地图的宽度")
    maze_height: int = Field(default=100, description="迷宫地图的高度")
    sq_tile_size: int = Field(default=32, description="每个方块的像素宽高")
    special_constraint: str = Field(
        default="", description="描述世界中可能存在的任何特殊约束的字符串"
    )
    tiles: list[list[dict]] = Field(default=[])
    address_tiles: dict[str, set] = Field(default=dict())
    collision_maze: list[list] = Field(default=[])

    @model_validator(mode="before")
    @classmethod
    def _init_maze(cls, values):
        maze_asset_path = values["maze_asset_path"]
        assert maze_asset_path
        maze_asset_path = Path(maze_asset_path)

        # 读取迷宫的元数据
        maze_matrix_path = maze_asset_path.joinpath("matrix")
        meta_info = read_json_file(maze_matrix_path.joinpath("maze_meta_info.json"))

        maze_width = int(meta_info["maze_width"])
        maze_height = int(meta_info["maze_height"])
        values["maze_width"] = maze_width
        values["maze_height"] = maze_height
        values["sq_tile_size"] = int(meta_info["sq_tile_size"])
        values["special_constraint"] = meta_info["special_constraint"]

        # 读取特殊方块数据
        blocks_folder = maze_matrix_path.joinpath("special_blocks")

        _wb = blocks_folder.joinpath("world_blocks.csv")
        wb_rows = read_csv_to_list(_wb, header=False)
        wb = wb_rows[0][-1]

        _sb = blocks_folder.joinpath("sector_blocks.csv")
        sb_rows = read_csv_to_list(_sb, header=False)
        sb_dict = dict()
        for i in sb_rows:
            sb_dict[i[0]] = i[-1]

        _ab = blocks_folder.joinpath("arena_blocks.csv")
        ab_rows = read_csv_to_list(_ab, header=False)
        ab_dict = dict()
        for i in ab_rows:
            ab_dict[i[0]] = i[-1]

        _gob = blocks_folder.joinpath("game_object_blocks.csv")
        gob_rows = read_csv_to_list(_gob, header=False)
        gob_dict = dict()
        for i in gob_rows:
            gob_dict[i[0]] = i[-1]

        _slb = blocks_folder.joinpath("spawning_location_blocks.csv")
        slb_rows = read_csv_to_list(_slb, header=False)
        slb_dict = dict()
        for i in slb_rows:
            slb_dict[i[0]] = i[-1]

        # [部分3] 读取迷宫矩阵
        maze_folder = maze_matrix_path.joinpath("maze")

        _cm = maze_folder.joinpath("collision_maze.csv")
        collision_maze_raw = read_csv_to_list(_cm, header=False)[0]
        _sm = maze_folder.joinpath("sector_maze.csv")
        sector_maze_raw = read_csv_to_list(_sm, header=False)[0]
        _am = maze_folder.joinpath("arena_maze.csv")
        arena_maze_raw = read_csv_to_list(_am, header=False)[0]
        _gom = maze_folder.joinpath("game_object_maze.csv")
        game_object_maze_raw = read_csv_to_list(_gom, header=False)[0]
        _slm = maze_folder.joinpath("spawning_location_maze.csv")
        spawning_location_maze_raw = read_csv_to_list(_slm, header=False)[0]

        # 加载迷宫。迷宫数据直接来自Tiled地图的JSON导出，格式为CSV。
        # 需要注意的是，数据并不是2D矩阵格式，而是单行矩阵，需要转换为二维格式。
        # 例如，[['0', '0', ... '25309', '0', ...], ['0', ...], ...]
        collision_maze = []
        sector_maze = []
        arena_maze = []
        game_object_maze = []
        spawning_location_maze = []
        for i in range(0, len(collision_maze_raw), maze_width):
            tw = maze_width
            collision_maze += [collision_maze_raw[i : i + tw]]
            sector_maze += [sector_maze_raw[i : i + tw]]
            arena_maze += [arena_maze_raw[i : i + tw]]
            game_object_maze += [game_object_maze_raw[i : i + tw]]
            spawning_location_maze += [spawning_location_maze_raw[i : i + tw]]
        values["collision_maze"] = collision_maze

        # 处理每个迷宫方块的详细信息
        tiles = []
        for i in range(maze_height):
            row = []
            for j in range(maze_width):
                tile_details = dict()
                tile_details["world"] = wb

                tile_details["sector"] = ""
                if sector_maze[i][j] in sb_dict:
                    tile_details["sector"] = sb_dict[sector_maze[i][j]]

                tile_details["arena"] = ""
                if arena_maze[i][j] in ab_dict:
                    tile_details["arena"] = ab_dict[arena_maze[i][j]]

                tile_details["game_object"] = ""
                if game_object_maze[i][j] in gob_dict:
                    tile_details["game_object"] = gob_dict[game_object_maze[i][j]]

                tile_details["spawning_location"] = ""
                if spawning_location_maze[i][j] in slb_dict:
                    tile_details["spawning_location"] = slb_dict[spawning_location_maze[i][j]]

                tile_details["collision"] = False
                if collision_maze[i][j] != "0":
                    tile_details["collision"] = True

                tile_details["events"] = set()

                row += [tile_details]
            tiles += [row]
        values["tiles"] = tiles

        # 设置每个游戏对象的事件
        for i in range(maze_height):
            for j in range(maze_width):
                if tiles[i][j]["game_object"]:
                    object_name = ":".join(
                        [tiles[i][j]["world"], tiles[i][j]["sector"], tiles[i][j]["arena"], tiles[i][j]["game_object"]]
                    )
                    go_event = (object_name, None, None, None)
                    tiles[i][j]["events"].add(go_event)

        # 反向索引：给定一个地址，返回所有属于该地址的坐标集合
        # 这个索引有助于优化查找角色移动路径
        address_tiles = dict()
        for i in range(maze_height):
            for j in range(maze_width):
                addresses = []
                if tiles[i][j]["sector"]:
                    add = f'{tiles[i][j]["world"]}:'
                    add += f'{tiles[i][j]["sector"]}'
                    addresses += [add]
                if tiles[i][j]["arena"]:
                    add = f'{tiles[i][j]["world"]}:'
                    add += f'{tiles[i][j]["sector"]}:'
                    add += f'{tiles[i][j]["arena"]}'
                    addresses += [add]
                if tiles[i][j]["game_object"]:
                    add = f'{tiles[i][j]["world"]}:'
                    add += f'{tiles[i][j]["sector"]}:'
                    add += f'{tiles[i][j]["arena"]}:'
                    add += f'{tiles[i][j]["game_object"]}'
                    addresses += [add]
                if tiles[i][j]["spawning_location"]:
                    add = f'<spawn_loc>{tiles[i][j]["spawning_location"]}'
                    addresses += [add]

                for add in addresses:
                    if add in address_tiles:
                        address_tiles[add].add((j, i))
                    else:
                        address_tiles[add] = set([(j, i)])
        values["address_tiles"] = address_tiles

        # 获取动作空间和观察空间
        values["action_space"] = get_action_space((maze_width, maze_height))
        values["observation_space"] = get_observation_space()
        return values

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, EnvObsValType], dict[str, Any]]:
        """重置环境并获取初始观测值
        返回结果对应于 `observation, info`
        """
        super().reset(seed=seed, options=options)

        obs = self._get_obs()

        return obs, {}

    def _get_obs(self) -> dict[str, EnvObsValType]:
        """获取观测值"""
        return {
            "collision_maze": self.get_collision_maze(),
            "tiles": self.tiles,
            "address_tiles": self.get_address_tiles(),
        }

    def observe(self, obs_params: Optional[EnvObsParams] = None) -> Any:
        """从环境中获取部分或完整的观测值"""
        obs_type = obs_params.obs_type if obs_params else EnvObsType.NONE
        if obs_type == EnvObsType.NONE:
            obs = self._get_obs()
        elif obs_type == EnvObsType.GET_TITLE:
            obs = self.access_tile(tile=obs_params.coord)
        elif obs_type == EnvObsType.TILE_PATH:
            obs = self.get_tile_path(tile=obs_params.coord, level=obs_params.level)
        elif obs_type == EnvObsType.TILE_NBR:
            obs = self.get_nearby_tiles(tile=obs_params.coord, vision_r=obs_params.vision_radius)
        return obs

    def step(self, action: EnvAction) -> tuple[dict[str, EnvObsValType], float, bool, bool, dict[str, Any]]:
        """执行动作并返回观测值
        返回结果对应于 `observation, reward, terminated, truncated, info`
        """
        terminated = False
        try:
            self._execute_env_action(action)
        except Exception:
            terminated = True

        obs = self._get_obs()

        ret = (obs, 1.0, terminated, False, {})
        return ret

    def _execute_env_action(self, action: EnvAction):
        """根据动作执行环境操作"""
        action_type = action.action_type
        if action_type == EnvActionType.NONE:
            pass
        elif action_type == EnvActionType.ADD_TILE_EVENT:
            self.add_event_from_tile(curr_event=action.event, tile=action.coord)
        elif action_type == EnvActionType.RM_TILE_EVENT:
            self.remove_event_from_tile(curr_event=action.event, tile=action.coord)
        elif action_type == EnvActionType.TURN_TILE_EVENT_IDLE:
            self.turn_event_from_tile_idle(curr_event=action.event, tile=action.coord)
        elif action_type == EnvActionType.RM_TITLE_SUB_EVENT:
            self.remove_subject_events_from_tile(subject=action.subject, tile=action.coord)

    def turn_coordinate_to_tile(self, px_coordinate: tuple[int, int]) -> tuple[int, int]:
        """
        将像素坐标转换为瓦片坐标
        """
        x = math.ceil(px_coordinate[0] / self.sq_tile_size)
        y = math.ceil(px_coordinate[1] / self.sq_tile_size)
        return x, y

    @mark_as_readable
    def get_collision_maze(self) -> list:
        """获取碰撞迷宫"""
        return self.collision_maze

    @mark_as_readable
    def get_address_tiles(self) -> dict:
        """获取地址瓦片"""
        return self.address_tiles

    @mark_as_readable
    def access_tile(self, tile: tuple[int, int]) -> dict:
        """
        返回指定坐标（x, y）位置的瓦片详细信息字典。

        输入:
          tile: 我们感兴趣的瓦片坐标，格式为 (x, y)。
        输出:
          指定瓦片的详细信息字典。
        示例输出:
          给定 (58, 9),
          self.tiles[9][58] = {'world': 'double studio',
                                'sector': 'double studio', 'arena': 'bedroom 2',
                                'game_object': 'bed', 'spawning_location': 'bedroom-2-a',
                                'collision': False,
                                'events': {('double studio:double studio:bedroom 2:bed', None, None)}}
        """
        x = tile[0]
        y = tile[1]
        return self.tiles[y][x]

    @mark_as_readable
    def get_tile_path(self, tile: tuple[int, int], level: str) -> str:
        """
        获取指定坐标的瓦片字符串地址，依据指定的层级（如：world、sector、arena、game object）。

        输入:
          tile: 我们感兴趣的瓦片坐标，格式为 (x, y)。
          level: 指定层级，可能值有：world、sector、arena、game object。
        输出:
          瓦片的字符串地址。
        示例输出:
          给定 tile=(58, 9)，level='arena'，
          返回: "double studio:double studio:bedroom 2"
        """
        x = tile[0]
        y = tile[1]
        tile = self.tiles[y][x]

        path = f"{tile['world']}"
        if level == "world":
            return path
        else:
            path += f":{tile['sector']}"

        if level == "sector":
            return path
        else:
            path += f":{tile['arena']}"

        if level == "arena":
            return path
        else:
            path += f":{tile['game_object']}"

        return path

    @mark_as_readable
    def get_nearby_tiles(self, tile: tuple[int, int], vision_r: int) -> list[tuple[int, int]]:
        """
        给定当前的坐标点和视距半径，返回一个在该半径范围内的所有方块坐标。
        注意，该实现是通过判断方块是否在一个方形区域内来确定视距范围。
        例如，对于视距半径vision_r，返回如下的方块(x)：
        x x x x x
        x x x x x
        x x P x x
        x x x x x
        x x x x x

        输入:
          tile: 当前方块的坐标，(x, y)格式。
          vision_r: 视距半径。
        输出:
          nearby_tiles: 一个列表，包含在视距半径内的方块坐标。
        """
        left_end = 0
        if tile[0] - vision_r > left_end:
            left_end = tile[0] - vision_r

        right_end = self.maze_width - 1
        if tile[0] + vision_r + 1 < right_end:
            right_end = tile[0] + vision_r + 1

        bottom_end = self.maze_height - 1
        if tile[1] + vision_r + 1 < bottom_end:
            bottom_end = tile[1] + vision_r + 1

        top_end = 0
        if tile[1] - vision_r > top_end:
            top_end = tile[1] - vision_r

        nearby_tiles = []
        for i in range(left_end, right_end):
            for j in range(top_end, bottom_end):
                nearby_tiles += [(i, j)]
        return nearby_tiles

    @mark_as_writeable
    def add_event_from_tile(self, curr_event: tuple[str], tile: tuple[int, int]) -> None:
        """
        向指定方块添加事件。

        输入:
          curr_event: 当前的事件元组。
            例如，('double studio:double studio:bedroom 2:bed', None, None)
          tile: 目标方块的坐标，(x, y)格式。
        输出:
          无
        """
        self.tiles[tile[1]][tile[0]]["events"].add(curr_event)

    @mark_as_writeable
    def remove_event_from_tile(self, curr_event: tuple[str], tile: tuple[int, int]) -> None:
        """
        从指定方块移除事件。

        输入:
          curr_event: 当前的事件元组。
            例如，('double studio:double studio:bedroom 2:bed', None, None)
          tile: 目标方块的坐标，(x, y)格式。
        输出:
          无
        """
        curr_tile_ev_cp = self.tiles[tile[1]][tile[0]]["events"].copy()
        for event in curr_tile_ev_cp:
            if event == curr_event:
                self.tiles[tile[1]][tile[0]]["events"].remove(event)

    @mark_as_writeable
    def turn_event_from_tile_idle(self, curr_event: tuple[str], tile: tuple[int, int]) -> None:
        """
        将指定方块的事件设为“闲置”状态。

        输入:
          curr_event: 当前的事件元组。
          tile: 目标方块的坐标，(x, y)格式。
        输出:
          无
        """
        curr_tile_ev_cp = self.tiles[tile[1]][tile[0]]["events"].copy()
        for event in curr_tile_ev_cp:
            if event == curr_event:
                self.tiles[tile[1]][tile[0]]["events"].remove(event)
                new_event = (event[0], None, None, None)
                self.tiles[tile[1]][tile[0]]["events"].add(new_event)

    @mark_as_writeable
    def remove_subject_events_from_tile(self, subject: str, tile: tuple[int, int]) -> None:
        """
        从指定方块移除包含特定主体的事件。

        输入:
          subject: 事件的主体，例如 "Isabella Rodriguez"
          tile: 目标方块的坐标，(x, y)格式。
        输出:
          无
        """
        curr_tile_ev_cp = self.tiles[tile[1]][tile[0]]["events"].copy()
        for event in curr_tile_ev_cp:
            if event[0] == subject:
                self.tiles[tile[1]][tile[0]]["events"].remove(event)
