# cue_contingency
Some thoughts for revising the Likelihood function and interpreting its current state:

Keep in mind that because there are only two cues and one of them HAS to be predictive on each trial, that if tau is the probability that the visual cue is VALID then the probability that the audio cue is VALID has to be 1 - tau.

2 key things to keep distinct:

The probability that a given cue is the predictive one (cue contingency---i.e., the outcome depends on the cue of this modality); and, 
The probability that a given cue is valid (cue validity---i.e., the probability that a predictive cue is correct).

The likelihood of a reward given a pairing of cues is given by the probability that the cue you *believe is predictive* is valid. In the case where cues are perfectly valid this gives a probability of 1 of being rewarded when the good cue is shown and a probability of 0 of being rewarded when the bad cue is shown. Then we can infer which of the cues is actually predictive by matching the outcome of a trial with the predictions of the cues... when they disagree. If they both make the same prediction (and this will be the correct prediction because one of them has to be predictive) then you can't tell which one is the predictive cue despite the 100% cue validity.

By introducing the possibility that cues aren't 100% valid, you can have cases where both of the cues make the same prediction and they're both wrong.

In the case where the cues aren't 100% valid, this gives a probability of being rewarded when the good Visual cue is shown equal to the reliability of the good Visual cue and a probability of being rewarded when the good Audio cue is shown equal to the reliability of the good Audio cue (from above, 1 - tau if tau is the reliability of the Visual cue).

