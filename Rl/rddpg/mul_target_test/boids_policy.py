# -*- coding: utf-8 -*-
# @Time : 2022/6/7 下午6:52
# @Author :  wangshulei
# @FileName: boids_policy.py
# @Software: PyCharm
import copy
import math

import numpy as np


class boids_policy:
    def __init__(self,
                 agent_number,
                 agent_com_size,
                 max_vel,
                 rule_1_distance=4.0,
                 rule_2_distance=4.0,
                 rule_3_distance=0.5):
        self.agent_number = agent_number
        self.max_vel = max_vel
        self.agent_com_size = agent_com_size
        self.rule_1_distance = rule_1_distance
        self.rule_2_distance = rule_2_distance
        self.rule_3_distance = rule_3_distance

    def one_agent_apply_boids_rules(self, now_agent_pos, all_pos, need_dist, now_agent_vel,
                                    all_vel, time_step, need_Hover=True):
        """
        单个智能体应用boids规则
        :param need_Hover: 直翼无人机不能够悬浮，这里需要对其设置盘旋的算法
        :param now_agent_vel: 当前智能体的速度
        :param now_agent_pos: 当前智能体的位置
        :param need_dist: 当前智能体与其他能通讯智能体的距离
        :param all_pos: 当前智能体与其他能通讯智能体的位置
        :param all_vel: 当前智能体与其他能通讯智能体的速度
        :param time_step: 当前运行步数
        :return:
        """
        # 该智能体观测范围内的距离
        all_agent_dists = [copy.deepcopy(need_dist)]
        rule1_acc = []
        now_agent_dist = all_agent_dists[0]  # 以及和他们的距离
        x_mean = now_agent_pos[0]
        y_mean = now_agent_pos[1]
        number_rule1_agent = 0
        for near_agent_index in range(len(now_agent_dist)):
            if now_agent_dist[near_agent_index] < self.rule_1_distance:
                x_mean += all_pos[near_agent_index][0]
                y_mean += all_pos[near_agent_index][1]
                number_rule1_agent += 1
        x_mean /= (number_rule1_agent + 1)
        y_mean /= (number_rule1_agent + 1)
        now_agent_acc_x = x_mean - now_agent_pos[0]
        now_agent_acc_y = y_mean - now_agent_pos[1]
        rule1_acc.append(np.array([now_agent_acc_x, now_agent_acc_y]))

        rule2_acc = []
        now_agent_dist = all_agent_dists[0]
        x_vel_mean = now_agent_vel[0]
        y_vel_mean = now_agent_vel[1]
        number_rule2_agent = 0
        for near_agent_index in range(len(now_agent_dist)):
            if now_agent_dist[near_agent_index] < self.rule_2_distance:
                x_vel_mean += all_vel[near_agent_index][0]
                y_vel_mean += all_vel[near_agent_index][1]
                number_rule2_agent += 1
        x_vel_mean /= (number_rule2_agent + 1)
        y_vel_mean /= (number_rule2_agent + 1)
        now_agent_acc_x = x_vel_mean - now_agent_vel[0]
        now_agent_acc_y = y_vel_mean - now_agent_vel[1]
        rule2_acc.append(np.array([now_agent_acc_x, now_agent_acc_y]))

        rule3_acc = []
        now_agent_dists = all_agent_dists[0]
        x_dist_mean = 0
        y_dist_mean = 0
        number_rule3_agent = 0
        for near_agent_index in range(len(now_agent_dist)):
            if now_agent_dists[near_agent_index] < self.rule_3_distance:
                # ? index(当前agent) -- near_agent_index(小于舒适距离的agent编号)
                x_dist_mean += (now_agent_pos[0] - all_pos[near_agent_index][0])
                y_dist_mean += (now_agent_pos[1] - all_pos[near_agent_index][1])
                number_rule3_agent += 1
        x_dist_mean /= (number_rule3_agent + 1)
        y_dist_mean /= (number_rule3_agent + 1)
        if x_dist_mean == 0 and y_dist_mean == 0:
            now_agent_acc_x = 0
            now_agent_acc_y = 0
        else:
            now_agent_acc_x = 1. / x_dist_mean
            now_agent_acc_y = 1. / y_dist_mean
        rule3_acc.append(np.array([now_agent_acc_x, now_agent_acc_y]))

        rule_acc = np.sum([rule1_acc, rule2_acc, rule3_acc], axis=0)
        rule_acc = self.apply_max_acc(rule_acc)

        action = []
        # 得到盘旋动作
        if need_Hover:
            points = []
            for i in range(125):
                ang = 2 * math.pi * i / 125
                points.append((math.cos(ang) * 3, math.sin(ang) * 3))
            Hover_action = np.array([0, np.clip(points[time_step % 125][0] - now_agent_pos[0], -0.3, 0.3),
                                     0, np.clip(points[time_step % 125][1] - now_agent_pos[1], -0.3, 0.3), 0])
        else:
            Hover_action = np.array([0, 0, 0, 0, 0])
        for rule_acc_one_agent in rule_acc:
            if time_step:
                action.append(np.array([0, 2 * rule_acc_one_agent[0], 0, 2 * rule_acc_one_agent[1], 0]) + Hover_action)
            else:
                action.append(np.array([0, 2 * rule_acc_one_agent[0], 0, 2 * rule_acc_one_agent[1], 0]))
        return action[0]

    def apply_max_acc(self, rule_all_agent_acc):
        all_acc = [np.sqrt(np.sum(np.square(agent_acc))) for agent_acc in rule_all_agent_acc]
        max_acc = max(all_acc)
        new_acc = []
        for rule_agent_acc in rule_all_agent_acc:
            if max_acc == 0:
                new_acc.append(rule_agent_acc)
            else:
                ratio = 1 / max_acc
                new_acc.append(ratio * rule_agent_acc * 0.3)
        return new_acc
