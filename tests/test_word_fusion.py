import unittest
from asr.timed import to_words
from asr.word_fusion import align_words,select_path,LanguageCounts


def word(text,start,score=-.1):
    return {'text':text,'start':start,'last_token_time':start,
            'end_estimated':start+.15,'end_is_estimated':True,
            'mean_logp':score,'min_logp':score,'log_probs':[score], 'token_indices':[0]}


class WordFusionTests(unittest.TestCase):
    def test_subtokens_and_approximate_word_end(self):
        result={'text':' DỊU THÔI','tokens':[' DỊ','U',' THÔI'],
                'timestamps':[.1,.2,.4],'ys_log_probs':[-.1,-.2,-.1],'duration_seconds':.7}
        words=to_words(result)
        self.assertEqual([w['text'] for w in words],['DỊU','THÔI'])
        self.assertEqual(words[0]['token_indices'],[0,1])
        self.assertEqual(words[0]['end_estimated'],.4)
        self.assertTrue(words[0]['end_is_estimated'])

    def test_many_to_one_spacing_and_missing_tail(self):
        g=[word('siro',.1),word('ho',.25),word('rồi',.5)]
        z=[word('siroho',.1)]
        cols=align_words(g,z)
        self.assertEqual(cols[0]['gipformer'],[0,1])
        self.assertEqual(cols[0]['zipformer'],[0])
        self.assertEqual([i for c in cols for i in c['gipformer']],[0,1,2])
        self.assertEqual([i for c in cols for i in c['zipformer']],[0])

    def test_numeric_value_cannot_be_mixed_from_two_paths(self):
        g=[word(t,i*.15) for i,t in enumerate('sáu mươi phút'.split())]
        z=[word('sáu',0),word('phút',.3)]
        cols=align_words(g,z)
        self.assertEqual(len(cols),1)
        self.assertIn('numeric_value',cols[0]['protected'])
        cols[0].update(p_gipformer=.99,agreement=False)
        self.assertEqual(select_path(cols,g,z)['text'],'sáu mươi phút')
        cols[0]['p_gipformer']=.01
        self.assertEqual(select_path(cols,g,z)['text'],'sáu phút')

    def test_different_words_in_one_sentence_can_use_different_models(self):
        g=[word('phong',0),word('sáu',1)]
        z=[word('phòng',0),word('hai',1)]
        cols=align_words(g,z)
        self.assertEqual(len(cols),2)
        for column,p in zip(cols,[.01,.99]):column.update(p_gipformer=p,agreement=False)
        result=select_path(cols,g,z)
        self.assertEqual(result['text'],'phòng sáu')
        self.assertEqual([r['chosen_source'] for r in result['decisions']],['zipformer','gipformer'])

    def test_repeated_words_follow_time_and_order(self):
        g=[word('minh',0),word('gọi',1),word('minh',2)]
        z=[word('minh',0),word('gọi',1),word('mình',2)]
        cols=align_words(g,z)
        self.assertEqual([(c['gipformer'],c['zipformer']) for c in cols],[([0],[0]),([1],[1]),([2],[2])])

    def test_identical_words_anchor_despite_long_decoder_delay(self):
        g=[word('ví',0),word('điện',1.1),word('tử',1.3)]
        z=[word('ví',0),word('điện',.2),word('tử',1.25)]
        cols=align_words(g,z)
        self.assertEqual([(c['gipformer'],c['zipformer']) for c in cols],[([0],[0]),([1],[1]),([2],[2])])
        for column in cols:column.update(p_gipformer=.5,agreement=True)
        self.assertEqual(select_path(cols,g,z)['text'],'ví điện tử')

    def test_ngram_probabilities_remain_bounded(self):
        language=LanguageCounts()
        for _ in range(20):language.add('bật đèn phòng khách'.split())
        self.assertLessEqual(language.score('bật đèn phòng khách'.split()),0)
        self.assertGreater(language.score('bật đèn phòng khách'.split()),language.score('bật đèn phong cách'.split()))

    def test_tokens_must_reconstruct_decoder_text(self):
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            to_words({'text':'đúng','tokens':[' sai'],'timestamps':[0],'ys_log_probs':[-.1],'duration_seconds':1})
