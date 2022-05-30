# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from reward import *
import json
from tqdm import tqdm, trange
import os
from torch.distributions import Multinomial

from layers import Scorer, VAE, Discriminator
from utils import TensorboardWriter

# labels for training the GAN part of the model
original_label = torch.tensor(1.0).cuda()
summary_label = torch.tensor(0.0).cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates the Thumbnail Selection model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Build Modules
        self.scorer = Scorer(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.vae = VAE(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.discriminator = Discriminator(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.model = nn.ModuleList([self.scorer, self.vae, self.discriminator])

        if self.config.mode == 'train':
            # Build Optimizers
            self.s_optimizer = optim.Adam(
                list(self.scorer.parameters()),
                lr=self.config.lr)
            self.e_optimizer = optim.Adam(
                list(self.vae.e_lstm.parameters()),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                list(self.vae.d_lstm.parameters()),
                lr=self.config.lr)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters()),
                lr=self.config.discriminator_lr)

            self.writer = TensorboardWriter(str(self.config.log_dir))

    def reconstruction_loss(self, h_origin, h_sum):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.norm(h_origin - h_sum, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)


    criterion = nn.MSELoss()

    def train(self):

        # Number of selected thumbnails
        num_of_picks = self.config.selected_thumbs

        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()

            recon_loss_history = []
            prior_loss_history = []
            g_loss_history = []
            e_loss_history = []
            d_loss_history = []
            c_original_loss_history = []
            c_summary_loss_history = []
            reward_history = []     
            rep_reward_history = []
            aes_reward_history = []

            # Training in batches of as many videos as the batch_size
            num_batches = int(len(self.train_loader)/self.config.batch_size)

            train_keys = self.train_loader.dataset.split['train_keys']

            # Baseline rewards for videos
            baselines = {key: 0. for key in train_keys}

            for batch in range(num_batches):                
                print(f'batch: {batch}')

                # ---- Train Encoder (eLSTM) ----#
                if self.config.verbose:
                    tqdm.write('Training Encoder...')

                iterator = iter(self.train_loader)
                self.e_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features, video_name, aes_scores_mean = next(iterator)

                    # [batch_size, seq_len, input_size]
                    # [seq_len, input_size]
                    image_features = image_features.view(-1, self.config.input_size)
                    image_features_ = Variable(image_features).cuda()
                    num_of_frames = image_features_.shape[0]

                    # [seq_len, 1, input_size]
                    original_features = image_features_.unsqueeze(1)
                    scores = self.scorer(original_features)  # [seq_len, 1]
                    s = scores.squeeze(1)  # [seq_len]
                    a = aes_scores_mean.squeeze(0).cuda()  # [seq_len]
                    sa = s*a  # [seq_len]

                    dist = Multinomial(num_of_picks, sa)
                    picks = dist.sample()  # binary tensor of size [seq_len] with the selected (1) and non-selected (0) frames

                    increase_picks = (torch.ones(num_of_frames)).cuda()
                    increase_picks = increase_picks + picks

                    weighted_scores = increase_picks.unsqueeze(1) * scores  # [seq_len, 1]
                    weighted_features = weighted_scores.view(-1, 1, 1) * original_features  # [seq_len, 1, input_size]

                    h_mu, h_log_variance, generated_features = self.vae(weighted_features)

                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)

                    rec_loss = self.reconstruction_loss(h_origin, h_sum)
                    prior_loss = self.prior_loss(h_mu, h_log_variance)

                    e_loss = rec_loss + prior_loss
                    e_loss = e_loss/self.config.batch_size
                    e_loss.backward()

                    prior_loss_history.append(prior_loss.data)
                    e_loss_history.append(e_loss.data)

                # Update e_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.vae.e_lstm.parameters(), self.config.clip)
                self.e_optimizer.step()

                # ---- Train Decoder (dLSTM) ----#
                if self.config.verbose:
                    tqdm.write('Training Decoder...')

                self.d_optimizer.zero_grad()
                iterator = iter(self.train_loader)
                for video in range(self.config.batch_size):
                    image_features, video_name, aes_scores_mean = next(iterator)

                    # [batch_size=1, seq_len, input_size]
                    # [seq_len, input_size]
                    image_features = image_features.view(-1, self.config.input_size)
                    image_features_ = Variable(image_features).cuda()
                    num_of_frames = image_features_.shape[0]

                    # [seq_len, 1, input_size]
                    original_features = image_features_.unsqueeze(1)
                    scores = self.scorer(original_features)  # [seq_len, 1]
                    s = scores.squeeze(1)  # [seq_len]
                    a = aes_scores_mean.squeeze(0).cuda()  # [seq_len]
                    sa = s * a  # [seq_len]

                    dist = Multinomial(num_of_picks, sa)
                    picks = dist.sample()  # binary tensor of size [seq_len] with the selected (1) and non-selected (0) frames

                    increase_picks = (torch.ones(num_of_frames)).cuda()
                    increase_picks = increase_picks + picks

                    weighted_scores = increase_picks.unsqueeze(1) * scores  # [seq_len]
                    weighted_features = weighted_scores.view(-1, 1, 1) * original_features  # [seq_len, 1, input_size]

                    h_mu, h_log_variance, generated_features = self.vae(weighted_features)

                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)

                    rec_loss = self.reconstruction_loss(h_origin, h_sum)
                    g_loss = self.criterion(sum_prob, original_label)

                    if self.config.verbose:
                        tqdm.write(f'recon loss {rec_loss.item():.3f}, g loss: {g_loss.item():.3f}')

                    d_loss = rec_loss + g_loss
                    d_loss = d_loss/self.config.batch_size
                    d_loss.backward()

                    recon_loss_history.append(rec_loss.data)
                    g_loss_history.append(g_loss.data)
                    d_loss_history.append(d_loss.data)

                # Update d_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.vae.d_lstm.parameters(), self.config.clip)
                self.d_optimizer.step()


                #---- Train Discriminator (cLSTM) ----#
                if self.config.verbose:
                    tqdm.write('Training Discriminator...')

                self.c_optimizer.zero_grad()
                iterator = iter(self.train_loader)
                for video in range(self.config.batch_size):
                    image_features, video_name, aes_scores_mean = next(iterator)

                    # [batch_size=1, seq_len, input_size]
                    # [seq_len, input_size]
                    image_features = image_features.view(-1, self.config.input_size)
                    image_features_ = Variable(image_features).cuda()
                    num_of_frames = image_features_.shape[0]

                    # Train with original loss
                    # [seq_len, 1, input_size]
                    original_features = image_features_.unsqueeze(1)
                    h_origin, original_prob = self.discriminator(original_features)
                    c_original_loss = self.criterion(original_prob, original_label)
                    c_original_loss = c_original_loss/self.config.batch_size
                    c_original_loss.backward()

                    # Train with summary loss
                    scores = self.scorer(original_features)  # [seq_len, 1]
                    s = scores.squeeze(1)  # [seq_len]
                    a = aes_scores_mean.squeeze(0).cuda()  # [seq_len]
                    sa = s * a  # [seq_len]

                    dist = Multinomial(num_of_picks, sa)
                    picks = dist.sample()  # binary tensor of size [seq_len] with the selected (1) and non-selected (0) frames

                    increase_picks = (torch.ones(num_of_frames)).cuda()
                    increase_picks = increase_picks + picks

                    weighted_scores = increase_picks.unsqueeze(1) * scores  # [seq_len, 1]
                    weighted_features = weighted_scores.view(-1, 1, 1) * original_features  # [seq_len, 1, input_size]

                    h_mu, h_log_variance, generated_features = self.vae(weighted_features)
                    h_sum, sum_prob = self.discriminator(generated_features.detach())
                    c_summary_loss = self.criterion(sum_prob, summary_label)
                    c_summary_loss = c_summary_loss/self.config.batch_size
                    c_summary_loss.backward()

                    c_original_loss_history.append(c_original_loss.data)
                    c_summary_loss_history.append(c_summary_loss.data)

                # Update c_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.clip)
                self.c_optimizer.step()


                # ---- Train Scorer ----#
                if self.config.verbose:
                    tqdm.write('Training Scorer...')

                self.s_optimizer.zero_grad()
                iterator = iter(self.train_loader)
                for video in range(self.config.batch_size):
                    image_features, video_name, aes_scores_mean = next(iterator)

                    # [batch_size=1, seq_len, input_size]
                    # [seq_len, input_size]
                    image_features = image_features.view(-1, self.config.input_size)
                    image_features_ = Variable(image_features).cuda()
                    num_of_frames = image_features_.shape[0]
                    
                    # [seq_len, 1, input_size]
                    original_features = image_features_.unsqueeze(1)
                    scores = self.scorer(original_features)  # [seq_len, 1]

                    s = scores.squeeze(1)  # [seq_len]
                    a = aes_scores_mean.squeeze(0).cuda()  # [seq_len]
                    sa = s * a  # [seq_len]

                    dist = Multinomial(num_of_picks, sa)

                    epis_rewards = []
                    epis_rep_rewards = []
                    epis_aes_rewards = []

                    sl = torch.tensor(0, dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)
                    s_loss = sl.clone()
                    for _ in range(self.config.n_episodes):
                        picks = dist.sample()  # binary tensor of size [seq_len] with the selected (1) and non-selected (0) frames

                        aes_reward = aesthetics_reward(aes_scores_mean.cuda(), picks, num_of_picks)

                        increase_picks = (torch.ones(num_of_frames)).cuda()
                        increase_picks = increase_picks + picks

                        weighted_scores = increase_picks.unsqueeze(1) * scores  # [seq_len, 1]
                        weighted_features = weighted_scores.view(-1, 1, 1) * original_features  # [seq_len, 1, input_size]

                        h_mu, h_log_variance, generated_features = self.vae(weighted_features)

                        h_origin, original_prob = self.discriminator(original_features)
                        h_sum, sum_prob = self.discriminator(generated_features)

                        rec_loss = self.reconstruction_loss(h_origin, h_sum)
                        rep_reward = 1 - rec_loss.item()

                        reward = (0.5 * rep_reward) + (0.5 * aes_reward)

                        log_probability = dist.log_prob(picks)
                        expected_reward = log_probability * (reward - baselines[video_name[0]])
                        s_loss -= expected_reward  # minimize negative expected reward

                        epis_rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                        epis_rep_rewards.append(torch.tensor([rep_reward], dtype=torch.float, device=device))
                        epis_aes_rewards.append(torch.tensor([aes_reward], dtype=torch.float, device=device))

                    s_loss.backward()
                    baselines[video_name[0]] = 0.9 * baselines[video_name[0]] + 0.1 * torch.mean(torch.stack(epis_rewards))

                    reward_mean = torch.mean(torch.stack(epis_rewards))
                    rep_reward_mean = torch.mean(torch.stack(epis_rep_rewards))
                    aes_reward_mean = torch.mean(torch.stack(epis_aes_rewards))

                    reward_history.append(reward_mean)
                    rep_reward_history.append(rep_reward_mean)
                    aes_reward_history.append(aes_reward_mean)

                # Update s_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), self.config.clip)
                self.s_optimizer.step()

                torch.cuda.empty_cache()

            recon_loss = torch.mean(torch.stack(recon_loss_history))
            prior_loss = torch.mean(torch.stack(prior_loss_history))
            g_loss = torch.mean(torch.stack(g_loss_history))
            e_loss = torch.mean(torch.stack(e_loss_history))
            d_loss = torch.mean(torch.stack(d_loss_history))
            c_original_loss = torch.mean(torch.stack(c_original_loss_history))
            c_summary_loss = torch.mean(torch.stack(c_summary_loss_history))

            reward_epoch = torch.mean(torch.stack(reward_history))
            rep_reward_epoch = torch.mean(torch.stack(rep_reward_history))
            aes_reward_epoch = torch.mean(torch.stack(aes_reward_history))

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(recon_loss, epoch_i, 'recon_loss_epoch')
            self.writer.update_loss(prior_loss, epoch_i, 'prior_loss_epoch')
            self.writer.update_loss(g_loss, epoch_i, 'g_loss_epoch')
            self.writer.update_loss(e_loss, epoch_i, 'e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_original_loss, epoch_i, 'c_original_loss_epoch')
            self.writer.update_loss(c_summary_loss, epoch_i, 'c_summary_loss_epoch')
            self.writer.update_loss(reward_epoch, epoch_i, 'reward_epoch')
            self.writer.update_loss(rep_reward_epoch, epoch_i, 'representativeness_reward_epoch')
            self.writer.update_loss(aes_reward_epoch, epoch_i, 'aesthetics_reward_epoch')

            # Save parameters at checkpoint
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            if self.config.verbose:
                tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)

    def evaluate(self, epoch_i):

        self.model.eval()

        num_of_picks = self.config.selected_thumbs
        out_dict = {}

        for image_features, video_name, aes_scores_mean in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, input_size]
            image_features = image_features.view(-1, self.config.input_size)
            image_features_ = Variable(image_features).cuda()
            num_of_frames = image_features_.shape[0]

            # [seq_len, 1, input_size]
            original_features = image_features_.unsqueeze(1)

            with torch.no_grad():
                scores = self.scorer(original_features)  # [seq_len, 1]
                s = scores.squeeze(1)  # [seq_len]
                a = aes_scores_mean.squeeze(0).cuda()  # [seq_len]
                sa = s * a  # [seq_len]

                dist = Multinomial(num_of_picks, sa)
                picks = dist.sample()  # binary tensor of size [seq_len] with the selected (1) and non-selected (0) frames

                increase_picks = (torch.ones(num_of_frames)).cuda()
                increase_picks = increase_picks + picks

                weighted_scores = increase_picks.unsqueeze(1) * scores  # [seq_len, 1]
                weighted_scores = weighted_scores.squeeze(1)
                weighted_scores = weighted_scores.cpu().numpy().tolist()

                out_dict[video_name] = weighted_scores

            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)
            score_save_path = self.config.score_dir.joinpath(
                f'{self.config.video_type}_{epoch_i}.json')
            with open(score_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            score_save_path.chmod(0o777)
            

if __name__ == '__main__':
    pass
