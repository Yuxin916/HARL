"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""

    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []

        # compute advantages
        if self.value_normalizer is not None:
            # gae / advantage -- 对比 OnPolicyCriticBufferEP.compute_returns
            # self.critic_buffer.returns： V网络的标签值 = GAE(step) + V网络(step)
            # self.critic_buffer.value_preds： V网络(step)

            # advantages [episode_length, 进程数量, 1] = Q - V
            advantages = self.critic_buffer.returns[:-1] - \
                         self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            # gae / advantage -- 对比 OnPolicyCriticBufferEP.compute_returns
            advantages = (self.critic_buffer.returns[:-1]
                          - self.critic_buffer.value_preds[:-1])

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            # advantage [n_rollout_threads, num_agents, 1]

        # actors更新
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(self.actor_buffer,
                                                               advantages.copy(),
                                                               self.num_agents,
                                                               self.state_type
                                                               )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            # 依次更新每个actor
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages.copy(),
                        "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[:, :, agent_id].copy(),
                        "FP",
                    )
                actor_train_infos.append(actor_train_info)

        # critic更新
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
