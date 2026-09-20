import unittest
from collections import Counter
from train.parakeet.focused.data import BalancedDraws,partition
from train.parakeet.focused.run import acceptable,selection
from contract.normalize import normalize

class FocusedProtocolTest(unittest.TestCase):
    def rows(self):
        return [dict(speaker_id=s, text=f'prompt {i}',mic=m,utterance_group=f'{s}_{i}',duration=1,utterance_id=f'{s}_{i}_{m}') for s in ('F01','M01','M04','F03') for i in range(60) for m in ('headMic','arrayMic')]
    def test_both_microphones_and_shared_prompts_stay_disjoint(self):
        train,dev,test=partition(self.rows())
        self.assertTrue(train and dev and test)
        for a,b in [(train,dev),(train,test),(dev,test)]:
            self.assertFalse({r['utterance_group'] for r in a}&{r['utterance_group'] for r in b})
            self.assertFalse({normalize(r['text']) for r in a}&{normalize(r['text']) for r in b})
    def test_split_membership_does_not_depend_on_input_order(self):
        for a,b in zip(partition(self.rows()),partition(list(reversed(self.rows())))):
            self.assertEqual({r['utterance_id'] for r in a},{r['utterance_id'] for r in b})
    def test_sampling_balances_speakers_despite_imbalance(self):
        rows=self.rows();rows=[r for r in rows if r['speaker_id']!='F01' or r['utterance_group']=='F01_0']
        draws=BalancedDraws(rows);counts=Counter(draws.draw()['speaker_id'] for _ in range(20000))
        for c in counts.values():self.assertLess(abs(c/20000-.25),.02)
    def test_selection_is_f01_first_and_guards_other_speakers(self):
        base={s:{'wer':.4} for s in ['F01','M01','F04','M04']}
        self.assertTrue(acceptable(base,base))
        self.assertFalse(acceptable({**base,'F04':{'wer':.43}},base))
        self.assertLess(selection({**base,'F01':{'wer':.3}}),selection(base))

if __name__=='__main__':unittest.main()
