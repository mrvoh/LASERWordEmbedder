from train import main as m_train
from test import main as m_test
from t_muse import main as muse_test
from train_muse import main as muse_train
from model.config import Config
import os
import json
if __name__ == "__main__":

	config = Config()
	for task in [
		'NER',
		# 'POS'
				]:
		config.set_pos_target(task)

		for lang in [
			# 'eng',
			# 'ned',
			'ger',
			# 'esp',
			# 'mixed'
					]:
			results = None
			config.langfolder = lang

			config.filename_train = os.path.join('parsed_data',
												 '{}_train_bio_bpe{}.txt'.format(lang, '1' if config.pos_target else ''))
			config.filename_dev = os.path.join('parsed_data',
												 '{}_valid_bio_bpe{}.txt'.format(lang, '1' if config.pos_target else ''))
			# for memory/speed
			config.batch_size = 32 if config.pos_target else 64
			# if lang == 'esp' or lang =='ned': # these languages have longer sequences
			# 	config.batch_size = int(config.batch_size / 2)
			# 	print(config.batch_size)

			m_train(config=config)
			results, _ = m_test(config=config)
			# muse_train(config=config, lang=lang)
			# results, _ = muse_test(results= results, config=config)
			out_path = os.path.join(config.results_folder, config.langfolder, config.subfolder, 'results.json')
			with open(out_path, 'w') as f:
				json.dump(results, f)