import h5py
import numpy as np
import sys
import programs


class ClevrDataLoader:
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('question_h5 was not provided.')
        if 'feature_h5' not in kwargs:
            raise ValueError('feature_h5 was not provided.')
        if 'vocab' not in kwargs:
            raise ValueError('vocab was not provided.')

        self.mode = kwargs.pop('mode', 'prefix')
        mode_choices = ['prefix', 'postfix']
        if self.mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % self.mode)

        feature_h5_path = kwargs.pop('feature_h5')
        print('Reading features from ', feature_h5_path)
        self.feature_h5 = h5py.File(feature_h5_path, 'r')

        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            print('Reading images from ', image_h5_path)
            self.image_h5 = h5py.File(image_h5_path, 'r')

        self.vocab = kwargs.pop('vocab')

        question_families = kwargs.pop('question_families', None)
        if question_families is not None:
            '''Use only the specified families'''
            all_families = np.asarray(question_families)[:, None]
            print(question_families)
            target_families = np.asarray(question_families)[:, None]
            mask = (all_families == target_families).any(axis=0)

        question_h5_path = kwargs.pop('question_h5')
        print('Reading questions from ', question_h5_path)
        self.question_h5 = h5py.File(question_h5_path, 'r')

        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(self.question_h5['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from

        self.max_samples = kwargs.pop('max_samples', None)

        '''Data from the question file is small, so read it all into memory'''
        print('Reading question data into memory')
        self.size = self.question_h5['questions'].shape[0]
        self.all_questions = self.question_h5['questions']
        self.all_image_idxs = self.question_h5['image_idxs']
        self.all_programs = None

        if 'programs' in self.question_h5:
            self.all_programs = self.question_h5['programs']

        self.all_answers = self.question_h5['answers']

    def __len__(self):
        if self.max_samples is None:
            return self.size
        else:
            return min(self.max_samples, self.size)

    def __enter__(self):
        pass

    def __exit__(self, ext, exv, trb):
        if ext is not None:
            print('An error has been caught...\n')
            print(exv)
        sys.exit(1)


def get_data(dataloader):
    question = dataloader.all_questions[0:]
    image_idx = dataloader.all_image_idxs[0:]
    answer = dataloader.all_answers[0:]
    
    program_seq = None

    if dataloader.all_programs is not None:
        program_seq = dataloader.all_programs[0:]

    image = None
    if dataloader.image_h5 is not None:
        image = dataloader.image_h5['images'][image_idx]

    features = []
    for i in image_idx:
        features.append(dataloader.feature_h5['features'][i])
    features = np.asarray(features, dtype=np.float32)

    program_json = None
    if program_seq is not None:
        program_json_seq = []
        for fn_idx in program_seq:
            for i in fn_idx:
                fn_str = dataloader.vocab['program_idx_to_token'][i]

                if fn_str == '<START>' or fn_str == '<END>':
                    continue

                fn = programs.str_to_function(fn_str)
                program_json_seq.append(fn)

        if dataloader.mode == 'prefix':
            program_json = programs.prefix_to_list(program_json_seq)
        elif dataloader.mode == 'postfix':
            program_json = programs.postfix_to_list(program_json_seq)

    return question, image, features, answer, program_seq, program_json
