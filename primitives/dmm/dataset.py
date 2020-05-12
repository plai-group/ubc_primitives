import torch
import pandas as pd
from torch.utils import data
import torchvision.transforms as transforms

__all__ = ('Dataset',)

class Dataset(data.Dataset):
    # Dataset
    def __init__(self, config, X, y, min_series_length, mode='TRAIN'):
        # Parse Model config
        self.mc = config
        self.batch_size = config.batch_size

        self.pre_process = transforms.Compose([transforms.ToTensor()])

        X_df, y_df, ids = self._get_trainable_df(X_df=X, y_df=y, min_series_length=min_series_length)

        # Transform long dfs to wide numpy
        assert type(X_df) == pd.core.frame.DataFrame
        assert type(y_df) == pd.core.frame.DataFrame
        assert all([(col in X_df) for col in ['unique_id', 'ds', 'x']])
        assert all([(col in y_df) for col in ['unique_id', 'ds', 'y']])

        X, y = self.long_to_wide(X_df, y_df)
        assert len(X)==len(y)
        assert X.shape[1]>=3

        # Exogenous variables
        unique_categories     = np.unique(X[:, 0])
        n_series              = len(unique_categories)
        mc["category_to_idx"] = dict((word, index) for index, word in enumerate(unique_categories))

        all_series = list(range(n_series))
        self.sort_key  = {'unique_id': [unique_idxs[i] for i in all_series],
                          'sort_key':  all_series}

        self.X = X
        self.y = y
        self.n_series = n_series


    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # Select sample
        u_id, label = self.sort_key['unique_id'][index]

        X_feature =
        y_lable   =

        # Load data and get label
        image = Image.open(img_path)
        image = self.pre_process(image)
        label = float(label)

        return image, label

    def shuffle_dataset(self, random_seed=1):
        """
        Return the examples in the dataset in order, or shuffled.
        """
        # Random Seed
        np.random.seed(random_seed)
        self.random_seed = random_seed
        shuffle = np.random.choice(self.n_series, self.n_series, replace=False)
        self.X = self.X[shuffle]
        self.y = self.y[shuffle]

        old_sort_key = self.sort_key['sort_key']
        old_unique_idxs = self.sort_key['unique_id']
        self.sort_key = {'unique_id': [old_unique_idxs[i] for i in shuffle],
                         'sort_key': [old_sort_key[i] for i in shuffle]}


    def _get_trainable_df(self, X_df, y_df, min_series_length):
        unique_counts = X_df.groupby('unique_id').count().reset_index()[['unique_id','ds']]
        ids = unique_counts[unique_counts['ds'] >= min_series_length]['unique_id'].unique()
        X_df = X_df[X_df['unique_id'].isin(ids)].reset_index(drop=True)
        y_df = y_df[y_df['unique_id'].isin(ids)].reset_index(drop=True)

        return X_df, y_df, ids

    def _long_to_wide(self, X_df, y_df):
        data = X_df.copy()
        data['y'] = y_df['y'].copy()
        sorted_ds = np.sort(data['ds'].unique())
        ds_map = {}
        for dmap, t in enumerate(sorted_ds):
            ds_map[t] = dmap
        data['ds_map'] = data['ds'].map(ds_map)
        data = data.sort_values(by=['ds_map','unique_id'])
        df_wide = data.pivot(index='unique_id', columns='ds_map')['y']

        x_unique = data[['unique_id', 'x']].groupby('unique_id').first()
        last_ds =  data[['unique_id', 'ds']].groupby('unique_id').last()
        assert len(x_unique)==len(data.unique_id.unique())
        df_wide['x'] = x_unique
        df_wide['last_ds'] = last_ds
        df_wide = df_wide.reset_index().rename_axis(None, axis=1)

        ds_cols = data.ds_map.unique().tolist()
        X = df_wide.filter(items=['unique_id', 'x', 'last_ds']).values
        y = df_wide.filter(items=ds_cols).values

        return X, y


  def __len__(self):
        # Total Number of samples
        return len(self.n_series)
