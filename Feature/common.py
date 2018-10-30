import argparse, datetime, os, logging, yaml, ujson, cPickle, time, pickle
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


def get_dairy_path(config, start_date, end_date):
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
                result_path.extend([os.path.join(r, f) for f in fn])
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
        saved_dict = defaultdict(list)

    while len(queue) and len(found_item)<len(item2search):
        node = queue.pop()

        if isinstance(node, dict):
            for k, v in node.items():
                if k in item2search:
                    saved_dict[k].append(v)
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
        saved_dict[item].append([None])

    return saved_dict


if __name__ == "__main__":
    args = parse_arguments()
    configs = yaml.load(args.config)

    data_paths = get_dairy_path(configs, "20181016", "20181016")
    print(data_paths)
    print( os.path.isdir(data_paths[0]))

    user_feature_names = [
        'app_version',
        'nl_topic256',
        'nl_topic64',
        'news_device_id',
        'os',
        'phone_model',
        'screen_height',
    ]

    news_feature_names = [
        'content_length',
        'country',
        'domain_category_level',
        'dup_flag',
        'enter_timestamp',
        'exp_scores',
        'evergreen_confidence',
        'first_occurrence_timestamp',
        'hub_timestamp',
        'index_type',
        'language',
        'last_timestamp',
        'topic256',
        'topic64',
        'ttl',
        'negative_feedback_keyword',
        'news_publish_time_diff_hour_desc', # available feature?
        'no_of_videos',
        'no_of_pictures',
        'partner',
        'pictures',
        'push_ctr',
        'quality',
        'recall_score',
        'response_timestamp',
        'sex'
    ]

    # filter active user data
    active_records = []

    with open(data_paths[0]) as f:
        for user_data in f.readlines():
            jason_data = " ".join(user_data.split()[3:])
            nested_dict = ujson.loads(jason_data)
            news_profile = nested_dict['news_profile']
            # user_profile = jason_data['user_profile']

            for news_id in news_profile.keys():
                label = news_profile[news_id]['label']

                if label:
                    active_records.append(nested_dict)
                    break

    print("number of active records: {}".format(len(active_records)))
    print(len(active_records))

    # extract features
    instances = []
    for user_data in active_records:
        news_profile = user_data['news_profile']
        user_profile = user_data['user_profile']

        user_feature = search_nested_dict_bfs(user_profile, user_feature_names, defaultdict(list))

        for news_id in news_profile:
            news_data_ = news_profile[news_id]
            instance_feature = search_nested_dict_bfs(news_data_, news_feature_names, user_feature.copy())
            instances.append(instance_feature)

    print(len(instances))

    cPickle.dump(instances, open('raw_feature.dat', 'wb'), True)


