

# from load_ISIC2018 import load_ISIC2018_GT

# def load_df(data_root):
#     return load_ISIC2018_GT(data_root)

# #%% data split for SSL

# np.random.seed(random_seed)
# train_indexes = np.random.permutation(len(train_df))
# val_indexes = np.random.permutation(len(val_df))
# test_indexes = np.arange(len(test_df))

# num_labels = int(len(train_df)*lbl_ratio)
# lbl_indexes = train_indexes[:num_labels]
# ulb_indexes = train_indexes[num_labels:]

# train_df.loc[ulb_indexes, 'labels'] = None
# train_df['labels'] = train_df['labels'].astype(object)  