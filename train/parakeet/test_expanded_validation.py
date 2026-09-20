import unittest
from train.parakeet.expanded_validation import partition


class ExpandedValidationTest(unittest.TestCase):
    def test_views_stay_together_and_order_does_not_change_split(self):
        rows = []
        for speaker in ('F01','M01','M03','M04','M05','F03'):
            for i in range(20):
                for mic in ('headMic','arrayMic'):
                    rows.append(dict(speaker_id=speaker, utterance_group=f'{speaker}_{i}',
                                     utterance_id=f'{speaker}_{i}_{mic}', mic=mic,
                                     text='one' if i<10 else 'two words'))
        train, dev, withheld = partition(rows)
        t2,d2,w2 = partition(list(reversed(rows)))
        ids = lambda rs: {r['utterance_id'] for r in rs}
        self.assertEqual(ids(train),ids(t2))
        self.assertEqual(ids(dev),ids(d2))
        self.assertEqual(ids(withheld),ids(w2))
        self.assertEqual(len(dev),20)
        self.assertEqual(len(withheld),40)
        self.assertTrue(all(r['mic']=='headMic' for r in dev))
        self.assertFalse({r['utterance_group'] for r in train}&{r['utterance_group'] for r in withheld})
        self.assertEqual(ids(train)|ids(withheld),ids(rows))
        self.assertFalse(any(r['speaker_id']=='F03' for r in dev))

    def test_array_only_group_is_not_dropped(self):
        rows=[dict(speaker_id='M05',utterance_group=str(i),utterance_id=str(i),mic='arrayMic',text='word') for i in range(10)]
        train,dev,heldout=partition(rows)
        self.assertEqual((len(train),len(dev),len(heldout)),(8,2,2))

    def test_rejects_inconsistent_microphone_labels(self):
        rows=[dict(speaker_id='M01',utterance_group='g',utterance_id=str(i),mic='headMic',text=t) for i,t in enumerate(['one','two'])]
        with self.assertRaises(ValueError): partition(rows)

if __name__ == '__main__': unittest.main()
