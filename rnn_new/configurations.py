def get_config_search_gru(ini_file=None):

    config = {}
        
    if ini_file is not None:
        with open(ini_file,'r') as in_file:
            for line in in_file:
                if len(line.strip()) == 0 or line.startswith('#'):
                    continue
                tokens = line.strip().split('=')
                assert len(tokens) ==2
                if tokens[1].isdigit():
                    val = int(tokens[1])
                else:    
                    val = tokens[1]
                config[tokens[0]] = val
    if 'workspace_dir' not in config:
        config['workspace_dir']= '/home/dhan/project_kws.0.1/split_sentence/rnn/'
    if 'root_dir' not in config:
        config['root_dir'] = '/home/dhan/project_kws.0.1/out_file/split_sentence/'
    
    if 'kws_class_dir' not in config:
        config['kws_class_dir'] = '/home/dhan/project_kws.0.1/kws_class/'

    if 'seq_len' not in config:
        config['seq_len'] = 30

    if 'nhids_num' not in config:
        config['nhids_num'] = 1000
    
    if 'embed_num' not in config:
        config['embed_num'] = 300

    if 'batch_size' not in config:
        config['batch_size'] =80 
    
    if 'sort_k_batches' not in config:
        config['sort_k_batches'] = 20

    if 'saveto_split' not in config:
        if 'workspace_dir' not in config:
            config['saveto_split'] = './train_model.npz'
        else:
            config['saveto_split'] = config['workspace_dir'] + 'train_model.npz'
    
    if 'saveto_split_best' not in config:
        if 'workspace_dir'  not in config:
            config['saveto_split_best']='./train_model_best.npz'
        else:
            config['saveto_split_best'] = config['workspace_dir'] + 'train_model_best.npz'
    config['train_file'] = config['root_dir']+"train.file"
    config['vocab_train_file']=config['kws_class_dir']+"doc/train_sent.txt"

    config['train_file_with_label'] = config['root_dir']+"train.file_with_label"
    
    config['valid_file'] = config['root_dir']+"dev.file"
    config['valid_out'] = config['root_dir']+"valid_out"
    config['test_file'] = config['root_dir']+"dev.file"
    config['test_out'] = config['root_dir']+"split_test_out"
    config['unk_id'] =1
    config['vector'] = config['root_dir']+"zh_vectors.pkl"
    config['embed'] = config['root_dir']+"embed.pkl"

    config['valid_split'] = config['root_dir']+"dev.file_with_label"
    if 'unk_token' not in config:
        config['unk_token']='unk'
    config['bos_token']='<S>'
    config['eos_token']='</S>'
    
    config['dropout'] = 0.7
    if 'vocab_size' not in config:
        config['vocab_size'] = 30000

    config['vocab'] =config['root_dir']+'vocab.pkl'
    
    config['finish_after']=10
    
    config['save_freq'] =100
    
    config['sample_freq'] = 10

    config['reload']=True
    
    config['hook_samples'] = 3
    return config
