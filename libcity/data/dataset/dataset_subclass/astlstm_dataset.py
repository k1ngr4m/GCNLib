import os
import numpy as np
from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.dataset import TrafficStateGridDataset
import sys

class ASTLSTMDataset(TrafficStateGridDataset):
    def __init__(self, config):
        super().__init__(config)
        self.points_per_hour = 3600 // self.time_intervals  # 每小时的时间片数
        self.len_closeness = self.config.get('len_closeness', 3)
        self.len_period = self.config.get('len_period', 4)
        self.len_trend = self.config.get('len_trend', 0)
        self.time_lag = self.config.get('time_lag', 6)
        self.TG = self.config.get('TG', 15)  # time interval(min)
        self.TG_in_one_day = self.config.get('TG_in_one_day', 48)  # time interval of one day
        self.TG_in_one_week = self.config.get('TG_in_one_week', 360)  # time interval of one week
        assert (self.len_closeness + self.len_period + self.len_trend > 0)

        self.interval_period = self.config.get('interval_period', 1)  # period的长度/天
        self.interval_trend = self.config.get('interval_trend', 7)  # trend的长度/天
        self.feature_name = {'X': 'float', 'y': 'float'}
        self.use_row_column = False
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.len_closeness) \
            + '_' + str(self.len_period) + '_' + str(self.len_trend) \
            + '_' + str(self.interval_period) + '_' + str(self.interval_trend) \
            + '_' + str(self.output_window) + '_' + str(self.train_rate) \
            + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) \
            + '_' + str(self.batch_size) + '_' + str(self.add_time_in_day) \
            + '_' + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample) \
            + '_' + str(self.time_lag) + '_' + str(self.TG) + '_' + str(self.TG_in_one_day) + '_' + str(
                self.TG_in_one_week)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _search_data(self, sequence_length, label_start_idx, num_for_predict, num_of_depend, units):
        # 与ASTGCNDataset中的_search_data方法相同
        """
                根据全局参数len_closeness/len_period/len_trend找到数据索引的位置

                Args:
                    sequence_length(int): 历史数据的总长度
                    label_start_idx(int): 预测开始的时间片的索引
                    num_for_predict(int): 预测的时间片序列长度
                    num_of_depend(int): len_trend/len_period/len_closeness
                    units(int): trend/period/closeness的长度(以小时为单位)

                Returns:
                    list: 起点-终点区间段的数组，list[(start_idx, end_idx)]
                """
        if self.points_per_hour < 0:
            raise ValueError("points_per_hour should be greater than 0!")
        if label_start_idx + num_for_predict > sequence_length:
            return None
        x_idx = []
        for i in range(1, num_of_depend + 1):
            # 从label_start_idx向左偏移，i是区间数，units*points_per_hour是区间长度(时间片为单位)
            start_idx = label_start_idx - self.points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))  # 每一段的长度是num_for_predict
            else:  # i越大越可能有问题，所以遇到错误直接范湖
                return None
        if len(x_idx) != num_of_depend:
            return None
        return x_idx[::-1]  # 倒序，因为原顺序是从右到左，倒序则从左至右

    def _get_sample_indices(self, data_sequence, label_start_idx):
        # 与ASTGCNDataset中的_get_sample_indices方法相同
        """
                根据全局参数len_closeness/len_period/len_trend找到数据预测目标数据
                段: [label_start_idx: label_start_idx+output_window)

                Args:
                    data_sequence(np.ndarray): 输入数据，shape: (len_time, ..., feature_dim)
                    label_start_idx(int): the first index of predicting target, 预测开始的时间片的索引

                Returns:
                    tuple: tuple contains:
                        trend_sample: 输入数据1, (len_trend * self.output_window, ..., feature_dim) \n
                        period_sample: 输入数据2, (len_period * self.output_window, ..., feature_dim) \n
                        closeness_sample: 输入数据3, (len_closeness * self.output_window, ..., feature_dim) \n
                        target: 输出数据, (self.output_window, ..., feature_dim)
                """
        trend_sample, period_sample, closeness_sample = None, None, None
        if label_start_idx + self.output_window > data_sequence.shape[0]:
            return trend_sample, period_sample, closeness_sample, None

        if self.len_trend > 0:
            trend_indices = self._search_data(data_sequence.shape[0], label_start_idx, self.output_window,
                                              self.len_trend, self.interval_trend * 24)
            if not trend_indices:
                return None, None, None, None
            # (len_trend * self.output_window, ..., feature_dim)
            trend_sample = np.concatenate([data_sequence[i: j] for i, j in trend_indices], axis=0)

        if self.len_period > 0:
            period_indices = self._search_data(data_sequence.shape[0], label_start_idx, self.output_window,
                                               self.len_period, self.interval_period * 24)
            if not period_indices:
                return None, None, None, None
            # (len_period * self.output_window, ..., feature_dim)
            period_sample = np.concatenate([data_sequence[i: j] for i, j in period_indices], axis=0)

        if self.len_closeness > 0:
            closeness_indices = self._search_data(data_sequence.shape[0], label_start_idx, self.output_window,
                                                  self.len_closeness, 1)
            if not closeness_indices:
                return None, None, None, None
            # (len_closeness * self.output_window, ..., feature_dim)
            closeness_sample = np.concatenate([data_sequence[i: j] for i, j in closeness_indices], axis=0)

        target = data_sequence[label_start_idx: label_start_idx + self.output_window]
        # (self.output_window, ..., feature_dim)
        return trend_sample, period_sample, closeness_sample, target

    # def _generate_input_data(self, df):
    #     # 首先调用ASTGCNDataset的_generate_input_data方法来处理数据
    #     sources, targets = self._generate_input_data_astgcn(df)
    #
    #     # 然后调用RESLSTMDataset的_generate_input_data方法来处理数据
    #     reslstm_X, reslstm_Y = self._generate_input_data_reslstm(df)
    #
    #     # 假设我们希望在第0维上匹配数组大小
    #     min_size = min(sources.shape[0], reslstm_X.shape[0])
    #
    #     # 截断较大的数组以匹配较小的数组
    #     sources = sources[:min_size]
    #     reslstm_X = reslstm_X[:min_size]
    #
    #     # 合并两个模型的输入和输出数据
    #     sources = np.concatenate((sources, reslstm_X), axis=-1)
    #     targets = np.concatenate((targets, reslstm_Y), axis=-1)
    #
    #     return sources, targets

    def _generate_input_data_astgcn(self, df):
        # 与ASTGCNDataset中的_generate_input_data方法相同
        """
               根据全局参数len_closeness/len_period/len_trend切分输入，产生模型需要的输入

               Args:
                   df(np.ndarray): 输入数据, shape: (len_time, ..., feature_dim)

               Returns:
                   tuple: tuple contains:
                       sources(np.ndarray): 模型输入数据, shape: (num_samples, Tw+Td+Th, ..., feature_dim) \n
                       targets(np.ndarray): 模型输出数据, shape: (num_samples, Tp, ..., feature_dim)
               """
        trend_samples, period_samples, closeness_samples, targets = [], [], [], []
        flag = 0
        for idx in range(df.shape[0]):
            sample = self._get_sample_indices(df, idx)
            if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
                continue
            flag = 1
            trend_sample, period_sample, closeness_sample, target = sample
            if self.len_trend > 0:
                trend_sample = np.expand_dims(trend_sample, axis=0)  # (1,Tw,N,F)
                trend_samples.append(trend_sample)
            if self.len_period > 0:
                period_sample = np.expand_dims(period_sample, axis=0)  # (1,Td,N,F)
                period_samples.append(period_sample)
            if self.len_closeness > 0:
                closeness_sample = np.expand_dims(closeness_sample, axis=0)  # (1,Th,N,F)
                closeness_samples.append(closeness_sample)
            target = np.expand_dims(target, axis=0)  # (1,Tp,N,F)
            targets.append(target)
        if flag == 0:
            self._logger.warning('Parameter len_closeness/len_period/len_trend is too large '
                                 'for the time range of the data!')
            sys.exit()
        sources = []
        if len(closeness_samples) > 0:
            closeness_samples = np.concatenate(closeness_samples, axis=0)  # (num_samples,Th,N,F)
            sources.append(closeness_samples)
            self._logger.info('closeness: ' + str(closeness_samples.shape))
        if len(period_samples) > 0:
            period_samples = np.concatenate(period_samples, axis=0)  # (num_samples,Td,N,F)
            sources.append(period_samples)
            self._logger.info('period: ' + str(period_samples.shape))
        if len(trend_samples) > 0:
            trend_samples = np.concatenate(trend_samples, axis=0)  # (num_samples,Tw,N,F)
            sources.append(trend_samples)
            self._logger.info('trend: ' + str(trend_samples.shape))
        sources = np.concatenate(sources, axis=1)  # (num_samples,Tw+Td+Th,N,F)
        targets = np.concatenate(targets, axis=0)  # (num_samples,Tp,N,F)
        return sources, targets

    def _generate_input_data(self, df):
        # 与RESLSTMDataset中的_generate_input_data方法相同
        data = np.array(df)
        X = np.zeros(
            (data.shape[0] - self.time_lag + 1 - self.output_window - self.TG_in_one_week, data.shape[1],
             (self.time_lag - 1) * 3, data.shape[-1]))
        Y = []

        for index in range(self.TG_in_one_week, data.shape[0] - self.time_lag + 1 - self.output_window):
            for i in range(self.num_nodes):
                temp = []
                for j in range(data.shape[-1]):
                    temp.append(data[index - self.TG_in_one_week: index + self.time_lag - 1 - self.TG_in_one_week, i,
                                j].tolist())
                    temp[-1].extend(
                        data[index - self.TG_in_one_day: index + self.time_lag - 1 - self.TG_in_one_day, i, j])
                    temp[-1].extend(data[index: index + self.time_lag - 1, i, j])

                for j in range(data.shape[-1]):
                    for k in range((self.time_lag - 1) * 3):
                        X[index - self.TG_in_one_week, i, k, j] = temp[j][k]

            Y.append(data[index + self.time_lag - 1: index + self.time_lag - 1 + self.output_window, :, :])

        X = X.swapaxes(1, 2)
        Y = np.array(Y)
        return X, Y

    def get_data_feature(self):
        # 返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        # feature_dim是输入数据的维度，output_dim是模型输出的维度,
        # len_closeness/len_period/len_trend分别是三段数据的长度
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "ext_dim": self.ext_dim,
                "len_closeness": self.len_closeness * self.output_window,
                "len_period": self.len_period * self.output_window,
                "len_trend": self.len_trend * self.output_window,
                "num_batches": self.num_batches}