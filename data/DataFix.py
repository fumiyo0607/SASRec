import pandas as pd
from collections import defaultdict

# dataset = 'Beauty'
minlen = 5

def creat_fixed_data(dataset, minlen):

    df = pd.read_table("../data/{}.txt".format(dataset), header=None, sep=' ')
    print('data num：{}'.format(len(df)))

    user_list  = df[0]
    item_list = df[1]
    user_list = list(reversed(user_list))
    item_list = list(reversed(item_list))

    User = defaultdict(list)

    for user, item in zip(user_list, item_list):
        User[user].append(item)

    cc = 0.0
    for u in User:
        cc += len(User[u])
    print('average sequence length: %.2f' % (cc / len(User)))

    ignore_user = []
    for u in User:
        if len(User[u]) < minlen:
            ignore_user.append(u)

    for u in ignore_user:
        User.pop(u)

    for u in User:
        items = User[u]
        items = items[0:5]

    User[u] = items 
    user_list = []
    item_list = []

    for u in User:
        items = User[u]
        for i in items:
            user_list.append(u)
            item_list.append(i)
    
    user_list = list(reversed(user_list))
    item_list = list(reversed(item_list))

    fixed_df = pd.DataFrame(
        data={
            'user_id': user_list,
            'item_id': item_list
        }
    )

    fixed_df.to_csv("../data/{}_min_{}.txt".format(dataset, minlen), header=None, sep=' ')

    print('user_num:{}'.format(len(User)))
    print('fied {} created ... ! '.format(dataset))

creat_fixed_data('Beauty', minlen)
creat_fixed_data('ml-1m', minlen)
creat_fixed_data('Video', minlen)
creat_fixed_data('Steam', minlen)