import argparse, datetime, os, logging, yaml, ujson, cPickle, time, pickle, pandas as pd, math, sys,numpy as np
from os import walk
from collections import defaultdict, Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def hot_encoding(df, cols):
    onehot_encoder = OneHotEncoder()
    dummies = []
    
    # get dummies
    for col in cols:
        df = pd.concat([df, pd.get_dummies(df[col], prefix = str(col))], axis = 1)
    
    # drop origin cols:
    df = df.drop(cols, axis = 1)
    return df

def parse_arguments():
    """Parse cli arguments
    :returns: args object

    """
    parser = argparse.ArgumentParser(description='Recommendation Algo Experiments')
    parser.add_argument("-c", "--config", type=argparse.FileType(mode='r'),
                        help="Config file")
    return parser.parse_args()


def get_dairy_path(config, start_date, end_date, filetype = '.csv'):
    """input: date ranges for dairy data
        output: return all files path in that range"""
    print('get_dairy_path')
    folder = os.path.join(os.getcwd(), config['data_folder'])
    result_path = []

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d')

    while start_date <= end_date:
        this_folder = os.path.join(folder, datetime.datetime.strftime(start_date, '%Y%m%d'))

        if os.path.isdir(this_folder):
            for r,d,fn in walk(this_folder):
                result_path.extend([os.path.join(r, f) for f in fn if f.endswith(filetype)])
        start_date += datetime.timedelta(days=1)

    if len(result_path)<1:
        logging.warnings('No dairy retrieved.')

    return result_path


def get_news_profile_from_dairy(filepath):
    record = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            # 'news_profiles' starts at index 3
            news_profiles = " ".join(line.split()[3:])
            news_profiles = ujson.loads(news_profiles)
            record.append(news_profiles)
    return record


def separate_clicked_record(filepaths):
    clicked_records = []
    _clicked_records = []

    for f in filepaths:
        dairy = get_news_profile_from_dairy(f)

        for user in dairy:

            active_flag = False

            for news_id in user['news_profile'].keys():
                if user['news_profile'][news_id].get('label') == 1:
                    active_flag = True
                    break

            if active_flag:
                clicked_records.append(user)
            else:
                _clicked_records.append(user)
    return clicked_records, _clicked_records


def search_nested_dict_bfs(root, item2search, saved_dict=None):
    queue = [root]
    found_item = set()

    if not saved_dict:
        saved_dict = {}

    while len(queue) and len(found_item)<len(item2search):
        node = queue.pop()

        if isinstance(node, dict):
            for k, v in node.items():
                if k in item2search:
                    if k in saved_dict.keys():
                        logging.warnings('overlapped feature names')
                    else:
                        saved_dict[k] = v
                        found_item.add(k)
                        if len(found_item) == len(item2search):
                            return saved_dict
                elif isinstance(v, dict):
                    queue.append(v)
        elif isinstance(node, list):
            for i in node:
                queue.append(i)

    not_found_item = [ i for i in item2search if i not in found_item]
    for item in not_found_item:
        saved_dict[item] = None

    return saved_dict


def extract_target_features_from_nested_jason(root, target_features, savedDict = None):
    found_feature = set()
    if not savedDict:
        savedDict = {}

    for key in root.keys():
        if key in target_features:
            if key in savedDict.keys():
                continue
                # logging.warnings("duplicate feature names")
            else:
                savedDict[key] = root[key]
                found_feature.add(key)

    not_found_feature = [a for a in target_features if a not in found_feature]

    for feat in not_found_feature:
        savedDict[feat] = None

    return savedDict


def extract_dairy_feature(active_records, feature_names):
    instances = []
    for user_data in active_records:
        news_profile_ = user_data['news_profile']
        user_profile_ = user_data['user_profile']

        user_feature = extract_target_features_from_nested_jason(user_profile_,
                                                                 feature_names['user_profile_feature_names'])
        user_feature['has_clicked_today'] = 0
        for news_entryid in news_profile_:
            news_data_ = news_profile_[news_entryid]
            news_feature_ = news_data_['news_feature']
            context_ = news_data_['context']
            label_ = news_data_['label']
            user_feature['has_clicked_today'] += label_

            user_feature_copy = user_feature.copy()
            user_feature_copy = extract_target_features_from_nested_jason(context_,
                                                                          feature_names['context_feature_names'],
                                                                          savedDict=user_feature_copy)
            user_feature_copy = extract_target_features_from_nested_jason(news_feature_,
                                                                          feature_names['news_feature_names'],
                                                                          savedDict=user_feature_copy)
            # get rank features for this user            
            user_feature_copy = extract_rank_features(user_feature_copy, \
                                                      ['response_time'])
            user_feature_copy['label'] = label_
            
            instances.append(user_feature_copy)
        
    instances = pd.DataFrame(instances)

    return instances


def extract_rank_features(df, col):
    rank_df = df[col]
    rank_df = rank_df.astype('int')
    rank_df = rank_df.rank(ascending = 1)
    new_columns = ['rank_' + i for i in rank_df.columns]
    df[new_columns] = rank_df
    return df


def extract_date_feature(df, dateCol):
    date = df[dateCol]
    weekday,hour=date.dt.weekday, date.dt.hour
    df['weekday'] = weekday
    df['hour'] = hour
    return df
    

def calculate_similarity(col_a, col_b, df):
    def calculate_sim(dict_a, dict_b, sim_type='euc'):
        sim = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for a_key in dict_a.keys():
            if a_key in dict_b.keys():
                if sim_type == 'euc':
                    sim += (a[a_key] - b[a_key]) ** 2
                elif sim_type == 'cos':
                    sim += a[a_key] * b[a_key]
                    norm_a += a[a_key] ** 2
                    norm_b += b[a_key] ** 2
                elif sim_type == 'abs':
                    sim += abs(a[a_key] - b[a_key])
                elif sim_type == 'jaccard':
                    sim += 1

        if sim:
            if sim_type == 'euc':
                sim = math.sqrt(sim)
            elif sim_type == 'jaccard':
                sim = sim / (len(dict_a.keys()) + len(dict_b.keys()) - sim)
        return sim

    merged_name = str(col_a) + '_' + str(col_b)
    euc_name = merged_name + '_euc_sim'
    cos_name = merged_name + '_cos_sim'
    abs_name = merged_name + '_abs_sim'
    jac_name = merged_name + '_jaccard_sim'
    similarities = {euc_name: [], cos_name: [], abs_name: [], jac_name:[]}

    for i in range(len(df)):
        a = df.iloc[i][col_a]
        b = df.iloc[i][col_b]

        # compute similarity
        if a and b:
            similarities[euc_name].append(calculate_sim(a, b, sim_type='euc'))
            similarities[cos_name].append(calculate_sim(a, b, sim_type='cos'))
            similarities[abs_name].append(calculate_sim(a, b, sim_type='abs'))
            similarities[jac_name].append(calculate_sim(a, b, sim_type='jaccard'))
        else:
            similarities[euc_name].append(None)
            similarities[cos_name].append(None)
            similarities[abs_name].append(None)
            similarities[jac_name].append(None)

    return similarities


def get_hardcoded_names():
    context_feature_names = [
        'news_publish_time_diff_hour_desc',
        'push_ctr',
        'rank_score',
        'recall_score',
        'recall_source',
        'response_timestamp'
        # 'channel', only push
    ]

    news_feature_names = [
         'category_v2_score',
         'country',
         'content_length',
         'crawler',
         'disgusting_scores',
         'domain',
         'dup_flag',
         'enter_timestamp',
         'enter_type',
         'entry_id',
         'exp_scores',
         'first_occurrence_timestamp',
         'hub_timestamp',
         'id',
         'image_mscv_scores',
         'image_nsfw_scores',
         'imageserver',
         'key_entities_v2',
         'key_entities_v2_hash',
         'key_entities',
         'keywords',
         'keywords_v2',
         'language',
         'location',
         'max_disgusting_scores',
         'news_id',
         'nlp_timestamp',
         'no_of_pictures',
         'original_url',
         'pictures',
         'publication_time',
         'push_source',
         'seed',
         'soure_num',
         'spam_word_count',
         'sub_category',
         'supervised_keywords',
         'target_country',
         'timestamp',
         'top_domain',
         'topic',
         'topic2048',
         'topic256',
         'topic64',
         'title',
         'title_keywords',
         'ttl',
         'url',
         'word_count'
    ]
    # 'author', feels not very important
    # 'thumbnail', more relevant to crawler
        # 'source_location', no non-empty entries
        # 'list_title', missing 99%
        # 'gallery_pictures_num', only FALSE value
        # 'hit_domain_blacklist', only FALSE value
        # 'head_addons', meaningless to me
        # 'body_addons', all null []
        # 'topic_v2', least overlap key value with user_topic feature
        # 'keywords_tag', no paird, and too much unique values
        # 'mannual_keywords', all false
        # 'supervised_keywords_v2', not many valuable values with user sv_keywords
        # 'supervised_keywords_v2_origin',not many non-zero values with user sv_keywords
        # 'category',
         # 'category_v2_cross',
         # 'new_type',
         # 'summary_length',
         # 'cdn_pictures',
         # 'news_state',
         # 'negative_feedback_category',
         # 'no_of_videos',
         # 'expiration_timestamp',
         # 'summary',
         # 'meta_keywords',
         # 'sex',
         # 'ads_state',
         # 'images',
         # 'combined_score', # std = 0
         # 'score',
         # 'evergreen_confidence',
         # 'relevant_entity',
         # 'quality',
         # 'sensitive_keywords',
         # 'index_type',
         # 'quality_entity_original_format',
         # 'domain_category_level',
         # 'negative_feedback_keyword',
         # 'topic_evergreen',
         # 'sanitized_html_length',
         # 'low_taste_keywords',
         # 'gallery_type',
         # 'quality_entity',
         # 'last_timestamp',
         # 'amp_url',
         # 'ads_location',
         # 'anti_spam_processed_time',
         # 'subscribe_tags',
         # 'news_location',
         # 'evergreen',
         # 'spelling_errors',
         # 'title_spam_word_count',
        # 'sub_sub_category',
        # 'sub_sub_sub_category',

    user_profile_feature_names = [
         'app_language',
         'app_version',
         'news_device_id',
         'manufacturer',
         'nl_category',
         'nl_domain',
         'nl_keywords',
         'nl_supervised_keywords',
         'nl_title_keywords',
         'nl_topic',
         'nl_topic2048',
         'nl_topic256',
         'nl_topic64',
         'nl_subcategory',
         'os',
         'phone_model',
         'product',
         'push_domain',
         'push_keywords',
         'push_topic',
         'push_topic2048',
         'push_topic256',
         'push_topic64',
         'push_supervised_keywords',
         'push_title_keywords',
         'screen_height',
         'screen_width',
         'system_language',
         'timezone'
        # 'push_subcategory', too much missing values
        # 'push_category', missing > 50%
        # 'appboy_id',
        # 'opera_id',
        # 'discover_id',
    ]

    rank_feature_names = [
        'news_publish_time_diff_hour_desc'
    ]
    
    feature_names = {'context_feature_names': context_feature_names,\
                     'news_feature_names': news_feature_names,\
                     'user_profile_feature_names': user_profile_feature_names,\
                     'rank_feature_names': rank_feature_names}

    return feature_names


def get_active_dairy_record(dairy_path, printInfo = True, dump = False):
    active_records = []

    with open(dairy_path) as f:
        for user_data in f.readlines():
            jason_data = " ".join(user_data.split()[3:])
            nested_dict = ujson.loads(jason_data)
            news_profile = nested_dict['news_profile']

            for news_id in news_profile.keys():
                label = news_profile[news_id]['label']

                if label:
                    active_records.append(nested_dict)
                    break

    if printInfo:
        print("number of active records: {}".format(len(active_records)))
        print(len(active_records))
    if dump:
        cPickle.dump(active_records, open('active_records.dat', 'wb'), True)
        print('active_records have been dumped')

    return active_records


def extract_version_feature(df, col):
    ctr_at_most_80 = ['1.0',
                      '2.1',
                      '2.2',
                      '2.3',
                      '2.4',
                      '3.0',
                      '3.2',
                      '3.3',
                      '4.1',
                      '4.2',
                      '4.3',
                      '4.4',
                      '44.1',
                      '44.2',
                      '44.6',
                      '45.0',
                      '45.1',
                      '46.0',
                      '46.1',
                      '46.3',
                      '47.0',
                      '47.1',
                      '47.2']
    ctr_at_least_80 = ['16.0',
                       '24.0',
                       '25.0',
                       '26.0',
                       '27.0',
                       '28.0',
                       '29.0',
                       '30.0',
                       '31.0',
                       '32.0',
                       '33.0',
                       '35.0',
                       '35.1',
                       '35.2',
                       '35.3',
                       '36.0',
                       '36.1',
                       '36.2']

    group1 = []
    group2 = []
    for app_version in df[col].values:
        app_version = '.'.join(app_version.split('.')[:2])
        g1 = False
        g2 = False

        if app_version in ctr_at_least_80:
            g1 = True
        elif app_version in ctr_at_most_80:
            g2 = True

        group1.append(g1)
        group2.append(g2)
    return group1, group2


# generate features pipeline.
def feature_pipeline(df):
    def calculate_all_similarities(paired_cols, df):
        new_df = df.copy()
        for c1, c2 in paired_cols:
            sim = calculate_similarity(c1, c2, df)
            new_df = new_df.join(pd.DataFrame(sim))

        # deal with subcategory, as one of them is a string not a dict
        sub_category_sim = []
        for a, b in zip(df['nl_subcategory'], df['sub_category']):
            if a and b:
                if b in a.keys():
                    sub_category_sim.append(a[b])
                else:
                    sub_category_sim.append(None)
            else:
                sub_category_sim.append(None)

        new_df['sub_category_sim'] = sub_category_sim
        return new_df

    # compute similarities against category, keywords, topics
    sim_pairs = [
        ("nl_category", "category_v2_score"),
        ("nl_keywords", "keywords"),
        ("nl_keywords", "keywords_v2"),
        ("nl_keywords", "push_keywords"),
        ("nl_supervised_keywords", "supervised_keywords"),
        ("nl_title_keywords", "title_keywords"),
        ("supervised_keywords", "push_supervised_keywords"),
        ("title_keywords", "push_title_keywords"),
        ("topic", "nl_topic"),
        ("topic2048", "push_topic2048"),
    ]
    newdf = calculate_all_similarities(sim_pairs, df)


    # drop unnecessary columns
    newdf = newdf.drop([
        "nl_category",
        "category_v2_score",
        "nl_keywords",
        "keywords",
        "keywords_v2",
        "push_keywords",
        "nl_supervised_keywords",
        "supervised_keywords",
        "nl_title_keywords",
        "title_keywords",
        "push_supervised_keywords",
        "push_title_keywords",
        "topic",
        "nl_topic",
        "push_topic2048",
        "topic2048",
        "push_topic256",
        "topic256",
        "push_topic64",
        "topic64",
        "nl_topic2048",
        "nl_topic256",
        "nl_topic64",
        "push_topic",
        "nl_subcategory",
        "sub_category",
        "dup_flag"
    ], axis=1)

    return newdf


def feature_pipeline2(df):
    # extract feature to two binary features
    df['app_version_ctr_80+'], df['app_version_ctr_80-'] = extract_version_feature(df, 'app_version')
    df = df.drop(['app_version'], axis = 1)
    return df


def get_all_feature_values(df, col):
    values = []

    for v in df[col]:
        if isinstance(v, dict):
            values.extend(v.keys())
        elif isinstance(v, list):
            values.extend(v)
        elif isinstance(v, str) or isinstance(v, unicode):
            values.append(v)
        else:
            print(v)
            raise ValueError(type(v))

    return values, set(values)


def count_non_values(df, col):
    count = 0

    for v in df[col]:
        if not v:
            count += 1
        elif v == '':
            count += 1
    return count, len(df)


def count_unqiue_values(values):
    unique = []

    for v in values:
        if v not in unique:
            unique.append(v)

    return unique, len(unique)


def extract_version_feature(df, col):
    ctr_at_most_80 = ['1.0',
                      '2.1',
                      '2.2',
                      '2.3',
                      '2.4',
                      '3.0',
                      '3.2',
                      '3.3',
                      '4.1',
                      '4.2',
                      '4.3',
                      '4.4',
                      '44.1',
                      '44.2',
                      '44.6',
                      '45.0',
                      '45.1',
                      '46.0',
                      '46.1',
                      '46.3',
                      '47.0',
                      '47.1',
                      '47.2']
    ctr_at_least_80 = ['16.0',
                       '24.0',
                       '25.0',
                       '26.0',
                       '27.0',
                       '28.0',
                       '29.0',
                       '30.0',
                       '31.0',
                       '32.0',
                       '33.0',
                       '35.0',
                       '35.1',
                       '35.2',
                       '35.3',
                       '36.0',
                       '36.1',
                       '36.2']

    group1 = []
    group2 = []
    for app_version in df[col].values:
        app_version = '.'.join(app_version.split('.')[:2])
        g1 = False
        g2 = False

        if app_version in ctr_at_least_80:
            g1 = True
        elif app_version in ctr_at_most_80:
            g2 = True

        group1.append(g1)
        group2.append(g2)
    return group1, group2


def filter_record(dairy_path, country_list, printInfo = True):
    # filter: >= 1 click, country in country_list, product = news
    active_records = []

    with open(dairy_path) as f:
        for user_data in f.readlines():
            jason_data = " ".join(user_data.split()[3:])
            nested_dict = ujson.loads(jason_data)
            user_profile = nested_dict['user_profile']
            
            if user_profile.get('product') != 'news':
                continue

            news_profile = nested_dict['news_profile']
            
            for news_id in news_profile.keys():
                label = news_profile[news_id]['label']
                country = news_profile[news_id]['news_feature']['country']
                if country in country_list and label:
                    active_records.append(nested_dict)
                    break
                
    if printInfo:
        print("number of active records: {}".format(len(active_records)))
    return active_records


if __name__ == "__main__":
    t = time.time()
    args = parse_arguments()
    configs = yaml.load(args.config)
    
    print('run works')

    # load dairy paths
#     data_paths = get_dairy_path(configs, "20180901", "20180901")
#     print(data_paths)

    # filter active users
#     active_records = get_active_dairy_record( data_paths[0] )

    # feature pipeline
#     df = cPickle.load(open('./Data/feature.dat', 'rb'))
#     df = feature_pipeline2(df)
#     cPickle.dump(df, open('./Data/feature1.dat', 'wb'), True)
#     print('dump app_version feature')
    # extract useful raw features
#     df = extract_dairy_feature(active_records, get_hardcoded_names())
#     cPickle.dump(df, open('./Data/raw_feature.dat', 'wb'), True)
#     print('dump raw_feature takes time: {}'.format(time.time() - t))

    # raw_df = cPickle.load(open('./Data/raw_feature.dat', 'rb'))
    #
    #
    # ## check which keywords are most relevant to a true label
    # co = 'push_keywords'
    # co_value = []
    # for label, s in zip(raw_df['label'], raw_df[co]):
    #     if label and s:
    #         co_value.extend(s.keys())
    #
    # a = Counter(co_value)
    #
    # # collect top 20 keyowrds:
    # top_20_keywords = [i for i, _ in a.most_common(20)]
    # top_20_keywords_feat = dict.fromkeys(top_20_keywords, [])
    #
    # # add top_20_keywords_feat
    # for p_keywords in raw_df['push_keywords']:
    #     hit_key = []
    #     if p_keywords:
    #         hit_key.extend(list(set(top_20_keywords) & set(p_keywords.keys())))
    #     rest_key = list(set(top_20_keywords) - set(hit_key))
    #
    #     for i in rest_key:
    #         top_20_keywords_feat[i].append(None)
    #
    # feat_df = raw_df.copy()
    # feat_df = feat_df.join(pd.DataFrame(top_20_keywords_feat))
    # feat_df.head()
