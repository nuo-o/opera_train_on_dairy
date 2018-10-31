import argparse, datetime, os, logging, yaml, ujson, cPickle, time, pickle, pandas as pd
from os import walk
from collections import defaultdict


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

    folder = config["data"]["folder"]
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


def read_files():
    raise NotImplementedError
    # read in all the files under the folder.
    # jason starts by user_profile:
    # extract corresponding keys
    # tidy up to list of dict
    # transform to panda dataframe


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


def clean_raw_dairy_2_df(active_records, feature_names):
    instances = []
    for user_data in active_records:
        news_profile_ = user_data['news_profile']
        user_profile_ = user_data['user_profile']

        user_feature = extract_target_features_from_nested_jason(user_profile_,
                                                                 feature_names['user_profile_feature_names'])

        for news_entryid in news_profile_:
            news_data_ = news_profile_[news_entryid]
            news_feature_ = news_data_['news_feature']
            context_ = news_data_['context']
            label_ = news_data_['label']

            instance_feature_ = user_feature.copy()
            instance_feature_ = extract_target_features_from_nested_jason(context_,
                                                                          feature_names['context_feature_names'],
                                                                          instance_feature_)
            instance_feature_ = extract_target_features_from_nested_jason(news_feature_,
                                                                          feature_names['news_feature_names'],
                                                                          instance_feature_)
            instance_feature_['label'] = label_
            instances.append(instance_feature_)

    df = pd.DataFrame(instances)

    return df


if __name__ == "__main__":
    t = time.time()
    args = parse_arguments()
    configs = yaml.load(args.config)

    data_paths = get_dairy_path(configs, "20181008", "20181008")

    print(data_paths)
    print( os.path.isdir(data_paths[0]))

    context_feature_names = [
        'recall_source',
        'push_ctr',
        'recall_score',
        'response_timestamp',
        'news_publish_time_diff_hour_desc',
        'channel',
        'rank_score'
    ]

    news_feature_names = [
         'content_length',
         'target_country',
         'topic256',
         'word_count',
         # 'new_type',
         'ttl',
         'keywords',
         # 'summary_length',
         'title',
         'max_disgusting_scores',
         # 'cdn_pictures',
         'location',
         'title_keywords',
         'hub_timestamp',
         'push_source',
         # 'news_state',
         'timestamp',
         # 'negative_feedback_category',
         'mannual_keywords',
         'nlp_timestamp',
         # 'no_of_videos',
         # 'expiration_timestamp',
         # 'summary',
         'disgusting_scores',
         'crawler',
         'domain',
         # 'meta_keywords',
         'sub_category',
         # 'sex',
         # 'ads_state',
         'key_entities',
         # 'images',
         'exp_scores',
         'enter_timestamp',
         'category',
         'body_addons',
         'pictures',
         'enter_type',
         'sub_sub_category',
         # 'combined_score', # std = 0
         'supervised_keywords_v2',
         # 'score',
         'keywords_tag',
         'thumbnail',
         'hit_domain_blacklist',
         # 'evergreen_confidence',
         'image_mscv_scores',
         'dup_flag',
         'language',
         'country',
         'category_v2_cross',
         'image_nsfw_scores',
         'no_of_pictures',
         'keywords_v2',
         # 'relevant_entity',
         # 'quality',
         'key_entities_v2',
         # 'sensitive_keywords',
         # 'index_type',
         'head_addons',
         'supervised_keywords_v2_origin',
         # 'quality_entity_original_format',
         'list_title',
         # 'domain_category_level',
         # 'negative_feedback_keyword',
         'topic2048',
         # 'topic_evergreen',
         # 'sanitized_html_length',
         # 'low_taste_keywords',
         'news_id',
         'category_v2_score',
         # 'gallery_type',
         'topic_v2',
         # 'quality_entity',
         # 'last_timestamp',
         'source_location',
         # 'amp_url',
         # 'ads_location',
         'topic',
         'seed',
         'entry_id',
         # 'anti_spam_processed_time',
         'id',
         'soure_num',
         'spam_word_count',
         'top_domain',
         'imageserver',
         'author',
         'topic64',
         'publication_time',
         # 'subscribe_tags',
         'first_occurrence_timestamp',
         # 'news_location',
         # 'evergreen',
         'supervised_keywords',
         # 'spelling_errors',
         'gallery_pictures_num',
         'url',
         'original_url',
         # 'title_spam_word_count',
         'sub_sub_sub_category',
         'key_entities_v2_hash'
    ]

    user_profile_feature_names = [
         'nl_category',
         'push_title_keywords',
         'push_supervised_keywords',
         # 'opera_id',
         # 'news_device_id',
         'push_keywords',
         'timezone',
         'push_topic2048',
         'phone_model',
         'nl_keywords',
         # 'discover_id',
         'push_subcategory',
         'nl_topic2048',
         'system_language',
         'nl_topic256',
         'app_version',
         'nl_supervised_keywords',
         'product',
         'screen_width',
         'push_topic256',
         'nl_subcategory',
         'push_topic64',
         'nl_title_keywords',
         'push_category',
         # 'appboy_id',
         'nl_topic64',
         'app_language',
         'manufacturer',
         'nl_domain',
         'push_topic',
         'push_domain',
         'nl_topic',
         'os',
         'screen_height']

    # filter active, only keep records that has at least one click per user
    active_records = []

    # for experiment: we only check one .csv file @@
    with open(data_paths[0]) as f:
        for user_data in f.readlines():
            jason_data = " ".join(user_data.split()[3:])
            nested_dict = ujson.loads(jason_data)
            news_profile = nested_dict['news_profile']

            for news_id in news_profile.keys():
                label = news_profile[news_id]['label']

                if label:
                    active_records.append(nested_dict)
                    break

    print("number of active records: {}".format(len(active_records)))
    print(len(active_records))

    cPickle.dump(active_records, open('active_records.dat', 'wb'), True)
    print('active_records have been dumped')

    # normalize dairy, extract features we need
    feature_names = {'context_feature_names': context_feature_names,\
                     'news_feature_names': news_feature_names,\
                     'user_profile_feature_names': user_profile_feature_names}
    df = clean_raw_dairy_2_df(active_records, feature_names)

    cPickle.dump(df, open('raw_feature.dat', 'wb'), True)
    print('write')

    print( time.time() - t)
    print('done')

