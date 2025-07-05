ALRIGHT!!! Let's GET INTO THIS THING!!!

### First steps, getting our bearings

To begin, we must first look back at Chatterbox.

Chatterbox was the first voice cloning model that I really had fun with, because for the first time it did something magical: at least for speaking, it was instantaneous, and made me truly gasp at how well it seemed to preserve some finer performance details.

Now, it is important to note that even so, the model usually changed things like prosody, pitch variations, and so on around in order to better fit the timbre.

One of my wishes has always been to get to try different timbres. I'm sure most singers, voice actors, and more have had this thought at one point or another. We're always taught that timbre is the *one* thing that, no matter how *much* we work at it, we just can't change because it's physical.

After all, how can we ever hope to change the way our skull resonates?

Things started changing however in recent years. And oh will we change it more today.

Let's begin.

The first step is *trying* to get Chatterbox, the original model, to convert a singing performance for us. See how far we are from the goal! This is a performance of Watch Me Fly by yours truly: [https://youtu.be/TPegbOryKF0](https://youtu.be/TPegbOryKF0). Listen until `4:00`.

Filled with anticipation, we use an additional trick to try to get Chatterbox to do singing: instead of loading the "target voice" as the actual target voice sample, we load *our own singing voice* into said slot, so that *hopefully* Chatterbox has a *singing* reference instead of the original talking reference. We *give* it the original performance directly, only swapping out the x-vector internally. This might work!!! My just before bed mind spoketh.

Alright, now start listening from `4:00` to `8:00` (don't listen past it yet).

...

It's! It's- *respectable!* Uh.

It has its great moments, that is true enough. Sometimes it gets the pitch right. However... it's far from a complete performance. It unfortunately only gets worse as the song continues. The section starting from `5:00` is especially hard to take in, with the seeming loss of energy, and then the shout. That shout.

However, we also look past the failures and see the incredible successes. *The content is still good*! The prosody, the lyrics. The prosody is not exactly like the original by any means given the model's original intentions were never to preserve it perfectly, but there are parts where it shines and gives very plausible prosodies. And the fact that none of the lyrics become scrambled is an amazing sign. It means that there is untapped potential here. But the glaring problem is still there. The precise pitch of the original performance is all but discarded, and the little bits that leak through correctly is sparse, and likely just an effect of the model getting it right by guessing, not by looking at the original performance.

After a dozen more attempts, we finally take a step back, take a deep breath, and realize that there's more work to be done.

It's time for the why.

### Why

Now that we've had some time to explore the capabilities of the model with our own ears, it's time to move to the next part. To understand the problem of *why* the original model seems to struggle immensely with pitch, even as it is able to *sort of* get half-way there, messing up the pitch greatly while the other parts seem to sound somewhat fine, we must first look closely at how the model works.

Let's first nail down what the original model's job is: **Take source audio, turn it into the same thing but in a different speaker's voice.**

We can immediately see how nebulous this goal is. However, the model recognizes this, and smartly breaks the problem down into pieces, each solveable on their own.

The core idea is to disentangle the various things that go into making someone's vocal recordings sound the way they do. We take the source recording (the recording we want to convert the tone of), and we extract:

1. `s3` tokens (`S3Tokenizer`)

The model listens to the source audio, and extracts *what* is being said. It represents this content as a sequence of discrete, universal "speech tokens." In text, these would be like the alphabet. Think of these as a phonetic or acoustic alphabet that is speaker-independent.

2. `x-vectors` (`CAMPPlus`)

The model listens to the source audio in the same manner, but the most important component of this model is that it does Temporal Aggregation (`StatsPool`) at the end and takes the final feature map of shape (Batch, Channels, Time) and computes the mean and standard deviation along the Time axis. This leaves only a single vector of shape (Batch, Channels * 2), with no time dimension. The phonetic sequence is gone, and only the global statistics remain. The core real world observation that this model is based on is that people's timbres do not change over time; thus, among all the different characteristics, the most prominent one that will be represented in this pooled output should be that.

To train this model, you would train it on a Speaker Identification task, giving the model a high loss when the x-vectors of two audio clips from the same person speaking is far apart and a low loss otherwise, and a similar idea for two audio clips from different people but in reverse.

*You might also note that once you have trained the timbre embedding model to get these x-vectors, you can then use a Speech Reconstruction task to train the `S3Tokenizer` mentioned above.*

This is because we should now theoretically have everything we need to rebuild the original voice clip from the encodings we have created. We have vectors that represent the *unchanging* parts of a person's voice. Meanwhile, we also have a new instance of an `S3Tokenizer` that's untrained, but is *supposed* to be encoding the *changing* parts of a person's voice (which is equivalent to encoding the *content* of the voice clip; the *content* is basically everything that's *not* timbre, right? Because content is the part we can control, and timbre is the only thing we cannot control)

So now we have, end-to-end, first the original voice clip, which the *encoder* model tries to extract the speaker-independent content from. Then, we pass that into a *decoder* model along with the calculated x-vector from the original voice clip (in practice we would quantize the output from the encoder using a "codebook", where the encoder's output is mapped to an index based on which vector from this codebook it is closest to, and then that *quantized* version of the vector from the codebook is passed along instead to prevent the model from cheating and subtly passing timbre information to the decoder; since the codebook is 6561 entries big, that's all the information you can pass, and there's no room to hide timbre information), and tell it to reconstruct the original MEL-spectrogram. The loss is how different the result is (plus some other things in new models), and once the models are trained we can get rid of that decoder model.

Wait.

DNO'T THROW AWAY THAT DECODER MODEL!!!

This is for training the tokenizer. However, after reading through this you might already have started to form an idea of how one could use this exact process in order to instead focus on teaching a decoder how to faithfully recreate audio! In fact you needn't change a single thing, because you already *have* such a model by the end of training that encoder.

As you can see, the encoder and the decoder together is the very heart of this model.

Now admittedly, at this point after all that model exposition, we still haven't found a way forward other than, "well, I guess just chuck musical samples at it now so that the encoder learns to represent it and the decoder learns to recreate it. Damn."

### Disentanglement

But wait. Wait. We need to take a step back. Let me pose an observation about singing (what we want) vs talking (what the model was trained on).

When **singing**, we prioritize pitch control over content-delivery and emotion-delivery at all costs. We want to "hit the note", have the sounds come out clean and on point!

Meanwhile, what do we do when talking? When **talking**, we prioritize content-delivery and emotion-delivery over precise pitch control.

Sometimes the two crosses over, but in the end the fundamental difference is there. There's that "soft" kind of difference that makes some logical sense... In order to achieve accurate pitch we have to sacrifice our natural speaking voice (which is the clearest in delivering content to another person almost all the time) somewhat, and it is probably not the most optimal way to communicate daily to try to talk in a sing-song voice at all times.

But that's not satisfying enough for me. "Kind of separate" is not good enough. Let's dig into the difference in deeper terms, in terms of pitch and content. That's when the difference becomes clear as day.

Unlike "content", **pitch is mathematical.** Two models can fight all day over their own embedding vectors to *represent* a specific human sound. "I think -ah should be represented by (1, 2, 1)." "No, I think -ah should be represented by (0.11, 3, 22)." Their task is not only to *interpret* content, but to **invent the language to represent that content.** Meanwhile, we already have a correct language to talk about pitch. 50hz is 50hz, 500hz is 500hz; it is objective.

So let's now think back to how this model architecture makes the whole monumental task approachable. The key was disentanglement. Different parts of speech have different characteristics, and in particular, we first disentangled timbre and content, because they had an inherent difference: timbre never changes, content always does (and encompasses everything else).

This is our way in.

This is our angle.

Do you see it?

Do you see the second point of disentanglement; the next step that has yet to be taken?

Hell yeah.

### Pitch vs content is the next natural point of disentanglement

We are in luck, because the original model's purpose was solely for normal everyday speaking, which means that pitch is *completely* foreign to the encoder for all it's concerned. Given its problems singing, it is likely that the encoder and decoder has both never even *encountered* singing before in its training dataset!

What's even more promising is that the encoder seems to be able to *ignore* pitch entirely! Listening back to that very first bad example of singing from before, we could clearly hear the content being hit, and the goal seemed so close yet so far. "If only the pitch was right-"

So now that we have determined the need to disentangle, the encoder is in a perfect state. It can infer the content from the source audio, while simultaneously completely ignoring the pitch. Thus, it is *already disentangled*. Now, we just need to provide the encoder with *another* conditioning signal, and hook up an encoder of *pitch* this time; it's the exact same end-to-end training process from before, but this time, we "flipped a valve".

This is incredibly promising, but before we can get to trying to implement this, we need to understand the decoder better. After all, neural network structures are not mutable, you can't typically change your inputs and their shapes after you finished training.

But at the same time, throwing all of the model's training away feels like wasted potential. Wasted work. The model is close, so close, and it just needs to now learn about pitch. We don't want to just throw away all that learning the decoder has already gotten on growls, sighs, gasps, puffs, and more, and we definitely don't want to throw away its ability to speak naturally. But if the conclusion of all these considerations ends up becoming, "well, I guess it's time to train a brand new model where the decoder can take both conditioning signals!" That would be wildly underwhelming.

Let us settle back down and look at the decoder. How does it do its job? Recreating audio is no joke, it is a very complicated process, and in fact the decoder probably has the toughest job out of all the component models.

### The Decoder Architecture

The decoder uses a **Conditional Flow Matching (CFM)** architecture.

The core of CFM is the following: Real data, such as human voices or images, can be thought of as living in a complex probability distribution in N dimensional space. You could imagine that there's a 'cloud' of probabilities that are denser where real data resides. We then have another, simpler probability distribution like the Gaussian distribution, which is noise, but also forms its own cloud.

CFM learns how to, starting from a random point in the noise distribution, move in the average direction towards the real data distribution. In more formal terms, it learns a vector field, which is a set of direction arrows at every single point in space, that defines the most efficient flow from the noise distribution to the data distribution.

The *way* it learns this "flow" (hence the name "flow matching") is through averaging and the Law of Large Numbers. For each training step, it:

1. Samples a real data point $x_1$ from the dataset.
2. Samples a noise vector $x_0$ from the Gaussian distribution (or any distribution for that matter).
3. Samples a random time $t$ between 0 and 1.
4. Calculates the interpolated point in between the two for time $t$: $x_t = (1-t)x_0 + tx_1$.
5. Trains the model: given the interpolated point $x_t$ which is the starting position and time $t$ as input, it must predict the target vector $(x_1 - x_0)$ (Note that it is not $(x_t - x_0)$, however; this is not an error. It is a subtle point, but the magnitude of the vector field is *velocity*, not direction. And velocity is constant here.).
6. Computes the loss based on the difference and use back-propagation to improve.

The key is that over millions of samples, even though, if you noticed from how we just picked any two examples from each distribution, each "path" we suggest might be "terrible" (e.g. crossing from one extreme edge of the noise region to another in the data region), but as it encounters more and more examples, the average of all these paths at any given point converges towards a smooth, coherent, and sensible direction.

To generate a new sample, you simply pick a random starting point from the noise distribution and follow the learned vector field.

This might seem similar to diffusion models, and indeed the principle is the same, but this is sort of a "condensed" version of that. In diffusion, each step is harder, and probabilistic, and the step you are predicting is a whole matrix of noise deltas. Meanwhile CFM is a more "canonical" or logical end result of solving this specific problem type. In effect, the core problem is learning how to go from one probability distribution's space towards another probability distribution's space. The most straightforward way to do it is to slowly learn the average vector field to move closer from one to the other. And the most computationally efficient way to do *that* is of course to utilize the LLN and do the subtraction. It's simple, elegant, much more direct in its approach, which is the beauty of it. `CausalConditionalCFM` is that flow-matching model here.

Both the noisy space and the real data space inhabit a joint probability distribution with 80*T random variables, and thus that many dimensions in the true probability distribution. The model, like we've discussed in the general case, then learns the best velocity for each value to move from one distribution to another.

The only *small* difference is that we have, in addition to the actual probabilistic regions encompassed by the input of (80, T) and the output of (80, T), a series of "conditioning" values that are also given to the decoder. Each of these conditioning values have the same dimensions as the noisy CFM MEL spectrograms, although it doesn't have to be so. The following table breaks down the entire input of `CausalConditionalCFM`:

| Vector Component | Source | Dimensions | Role / Conditioning Provided |
|------------------|--------|------------|------------------------------|
| Noisy Mel `x` | Conditional Flow-Matching Process | (80, T) | "What am I currently looking at? (The state to be refined)" |
| Prosodic and phonetic context `mu` | *From user's performance.* `s3_tokens`, through an adapter `encoder_proj` projecting it from 512-dims to 80-dims | (80, T) | "What should the content and rhythm be?" |
| Timbre `spks` | `x-vector`, through an adapter `spk_embed_affine_layer` projecting it from 192-dims to 80-dims | (80, 1) repeated to (80, T) | "What's the target timbre?" |
| Prompt `cond=prompt_feat` | *From target's performance.* The original target's performance converted into a MEL spectrogram format. | (80, ?), with the rest filled in with zeroes if not long enough | "Here's a concrete example of the target voice." |
| Total |  | (320, T) |  |

Here comes our problem. With the model's current inputs, **it doesn't have any information on the pitch of the original performance!** It's almost unreasonable to expect the model to perform well with singing, because the best it can do is just to guess from the prosody and phonetic content what a likely pitch would be for someone with the provided timbre to naturally speak that content out loud. This is exactly what we have expected. So right, `x`, `mu`, `spks`, ...

`prompt_feat`? That's new...

I was originally completely unsure as to what this was meant to be. Looking behind the scenes, it appears that this input is created by first taking the target's performance sample (from which the original model extracts an x-vector as well) and converting it into a MEL spectrogram. This MEL spectrogram is then placed into `prompt_feat` wholesale. My understanding is that this gives the decoder model one last signal: since our desire is to change the timbre completely, by having a real example of the "target's" voice directly *in* there as a separate conditioning signal as well, the model can simply "take" a lot of patterns, ways of speech, and more from this signal instead of having to rely solely on the information contained inside the x-vectors.

However, in my testing I believe this signal ended up being quite weak in the final model. You can try this experiment out yourself:

1. Load in a completely blank audio file as the target voice.
2. Then, swap out the x-vector (`speech_feat` inside the `ref_dict`) with a precomputed one from a desired voice file.
3. Pass in your own voice file and run the model as usual.

You'll find that the results are pretty similar.

This is actually a good thing for the model, because one of the hardest and worst problems in training a model like this is for the model to find "shortcuts", and thus miss out on learning a more fundamental and deep understanding of the problem space because it can get a pretty good loss pretty quickly by "cheating".

In this case, if the model ended up relying on this specific conditioning signal, `speech_feat`/`prompt_feat` too much because it's too useful, the model would have completely missed out on the opportunity to learn for itself *why* that person sounds that way. To learn how the x-vectors, which are embedding for the voice type the person has, actually translates when they talk. Instead, they might just use tricks to pull those patterns from the `prompt_feat`, take steps towards those patterns in the real data space instead without ever knowing *why* they were taking those steps. In essence, one approach would be trying to learn how the human vocal cord with that specific physical shape could produce those s3 tokens representing speech content, while the other is just shaping the noise based on the s3 tokens and the example until it looked "sort of similar", so that the loss function was happy.

It's a good thing then that it seems like this signal didn't end up being easier to use than the x-vector signal; my hypothesis is that it was just too hard to use for the model.

Human speech is incredibly variable and can produce a wide variety of sounds: the `prompt_feat` is just a *very* small slice from that, a short clip that may or may not have any of the important varieties of sounds that a model might need, even when they are dealing with matching content. To make use of this signal, the model must also learn to understand a *second* MEL spectrogram in addition to the noisy one the model is working on, which is a far more monumental task (learn an internal representation and understanding of an entire raw MEL spectrogram) than the fully-trained concise-and-mathematical x-vectors.

This signal is the perfect spot to take over. Instead of providing a `prompt_feat`, we will instead provide the signal we had been talking about. An 80-dimensional vector produced by another encoder model of our own design, specifically meant to capture pitch variety, as opposed to the unpitched s3 tokens.

Let's revise that table:

| Vector Component | Source | Dimensions | Role / Conditioning Provided |
|------------------|--------|------------|------------------------------|
| Noisy Mel `x` | Conditional Flow-Matching Process | (80, T) | "What am I currently looking at? (The state to be refined)" |
| Prosodic and phonetic context `mu` | *From user's performance.* `s3_tokens`, through an adapter `encoder_proj` projecting it from 512-dims to 80-dims | (80, T) | "What should the content and rhythm be?" |
| Timbre `spks` | `x-vector`, through an adapter `spk_embed_affine_layer` projecting it from 192-dims to 80-dims | (80, 1) repeated to (80, T) | "What's the target timbre?" |
| ~~Prompt `cond=prompt_feat`~~<br><br>**Pitch context** | ~~*From target's performance.* The original target's performance converted into a MEL spectrogram format.~~<br><br>***From user's performance.* Some sort of embedding of how the pitch is changing in the user's performance.** | ~~(80, ?), with the rest filled in with zeroes if not long enough~~<br><br>**(80, T)** | ~~"Here's a concrete example of the target voice."~~<br><br>**"What should the pitch be?"** |
| Total |  | (320, T) |  |

So how should we design such a model?

Actually, before we even get there, we need to answer an earlier question... how do we even calculate the inputs for said model? There are countless possibilities for how we could give some sort of input with pitch information. One that I initially thought of and quickly discarded was the idea of using some sort of pitch tracker. You'll understand immediately why if you've had to work with pitch correction before: even the best ones found in software like Melodyne make mistakes constantly.

Why not just pass in the MEL frames directly? Right, that would create the issue of giving the model a shortcut.

What we need is some sort of input that depicts the pitch-related information about the performance without including any information about the timbre. But that's a different task entirely, how would such a representation even look like!? I mean, isn't pitch-related performance details inherent *tied* to the person's timbre?

Well...

### F0

This is already a widely researched and textbook-heavy area, and you might already know some of this, but the core idea is that the human voice operates in a source-filter model.

First, when we speak or sing, air from our lungs pushes up against our vocal folds, causing pressure to build until it forces these folds apart, releasing a puff of air. But then the folds are brought back together by their elasticity, and so the cycle repeats very quickly.

This creates a sine wave, and the rate of this vibration is the Fundamental Frequency (F0). This is what we perceive as the pitch of the sound.

The key difference between how our voices sound however, is in the overtones. As the name implies, "Source-Filter Theory" is named that way because it breaks down the way we produce sound into these two stages.

First we have the source, which is our vocal folds producing that F0 wave. But even at this stage, we already have a source of difference in timbre, because our vocal folds don't just produce a pure sine wave. It also produces a sine wave at 2x the frequency. And also one at 3x, and 4x, and so on and so forth, each with a different intensity; this is the harmonic series, and the result is that while we can perceive the sound as F0 or at least at a pitch related to it, the same pitch *sounds* different coming from say a violin vs our voice. It is also the first step of what distinguishes our timbre (and is obviously something we can never hope to change on the fly).

Then comes the "filter" part. The voice, once produced in our vocal folds, has to travel out to be heard! It goes through our throat, our mouths, sometimes our noses; and each step of the way this original signal then gets filtered and changed as well. It amplifies certain bands of harmonics and dampens some others. Thus the filter part also has two components, a "changeable filter" (vocal tract, mouth shape, tongue position, etc.) which you *can* control, and then an "unchangeable filter" (our physical anatomy, which determines our head shape, how our noses are shaped, etc, which are resonate and change the intensity of the different harmonics).

And importantly, **this is how our speech *always* works, not *just* for singing!** When talking, when growling, when huffing. Always. (Though in the case of growling and such the source also includes a noise generator.)

Perhaps you can already see the insight to be gained from all of this biology.

*"The key difference between how our voices sound however, is in the overtones."*

**But our Fundamental Frequencies are always the same.**

When I try to talk or sing at C3, and you do the same, it might sound different, but in the end, that *source*, it's *always* the same C3!

And there's a model designed specifically to infer said F0, as this is a widely researched area: CREPE.

It is perhaps even the only way I can think of right now that you could possibly fulfill the seemingly self-contradicting requirement of, "depict the pitch-related information about a performance without including any information about the timbre (aka, harmonics pitch profile)". Models solving this family of problems must learn to get rid of speaker-identifying data because such data is *noise*. Timbre has the habit of concealing the fundamental frequency sometimes, so a model interested only in guessing at the F0 of that "Source" part of the model will not have a liking for it.

It is just *perfect* then that CREPE has a hidden state layer right before it's final logit layer which we can use as a pitch embedding.

So we just shove the pitch embeddings in and job done, right?

Well, not quite...

### Replicating Performance != Copying Over Pitch Fluctuations

If I gave you a piece of sheet music, and then handed you a list of instructions like "At time 1:01, move your pitch immediately up to a C3, and then start oscillating there at a rate of 10 times per second before letting it fall back down to a B3", you will stare at me with a glare and then let out your frustration to me at being asked to do something so monumentally stupid. But this exactly what we would be telling the model to do if we just chucked pitch embeddings at it. You'll know what I mean if you've had a chance to play around with a vocoder: it's a very specific sound, but it's definitely not human voice.

The insight below the intuition is that performances aren't *just* about pitch necessarily, although it is an important part of it. But what it really is is communicating "intention". Each thing we do in a performance, whether that be in regular singing or in a theatre, whether that be having smalltalk with people in work or class or shouting for a rock concert, is done with intent first, and pitch later.

The example I gave at the start is a vibrato, as you might have guessed, and the next words you might have said would probably be "That's just a vibrato right?" And you'd be right. But you'd also be right to boil it down to that, because there's a million different ways to do a vibrato.

"A million"? Isn't that a bit of a stretch?

It might seem like it, but it is true. For just a baseline, keep in mind that a vibrato could have different widths between each oscillation. Now we can do some napkin math, and if we put said duration in between oscillations at 16 different widths, and then add dynamics on top of that, you can quickly see that, actually, there's a *lot* of different ways you can do a vibrato for different effect.

But we're just *really* good at controlling our pitch in order to get that *exact* one we really like, and do it the same way each time. How? Because that *specific* vibrato has meaning.

For example, that classic little lilt you might do at the end of lines in songs, there's a very specific feel that comes with it, right? That's why you might be able to replicate that exact little drop in dynamics and that speed up in vibrato speed that sounds so good each time perfectly.

Now imagine that sort of idea but at a fine level: Even patterns in between and within individual MEL frames (which happen 50 times per second) have such a thing going for it. The model would be clueless and apply the pitch fluctuations roughly to the resulting performance, but that will sound incredibly robotic because while we're not changing the pitch in the way that a vocoder might or which an autotune software might want to do, we *are* changing pitch. What we are doing is *completely* swapping out that timbre harmonic series, which isn't just one pitch, but C1, C2, C3, C4, C5, and so on and so forth, modifying *each* of those frequencies, along with *additional* things like noise and grunts and even more, all to fit that "intent" we were talking about. At this micro-level, brute-forcing the work of changing *all of that* into a new timbre while keeping the "intent" (the desired performance effect that produces that specific recognition of "that's that end-of-a-phrase vibrato pattern!") would be an *astonomically* difficult task.

Our decoder model already has enough of a hard job: it has to recreate the entire actual performance from just a bunch of s3 tokens, a single x-vector, and now a bunch of pitch embeddings. We don't need to make its task astonomically harder by forcing it to allocate a big, massive chunk of its capabilities learning all these details about pitch.

### The New Model

So we do not do that. Instead, we design a model that is hopefully inherently amazing at trying to gain this sort of understanding. And an early hint at what the final model will entail is the following: We have a bunch of seemingly mathematical and arbitrary things that don't have much meaning on their own, but that people have inherently attributed great meaning to when taken together as a whole.

CREPE operates at 16khz, and with a step size of 10ms (a reasonable lower bound for CREPE to be able to infer interesting information,), we get a total of (16000 / 160 = 100) CREPE frames per second.
The MEL generation operates at 24khz, and with a hop size of 480 samples, so we get a total of (24000 / 480 = 50) MEL frames per second.
There is an opportunity here. For every MEL frame, we can provide the MEL generator with information hidden within two highly information-dense CREPE vectors.

We have two CREPE frames. Each frame is a 256-dimensional vector that is a rich representation of the current vocal pitch (and perhaps any surrounding pitches as well).
Given this, we need to learn an embedding for *the various ways someone could exercise their voice to convey human intent.*
Here's a small view of just a few of those techniques we hope the model will learn:

**Vibrato:** A musical vibrato will manifest as a periodic oscillation in the CREPE embeddings across the two time steps. Thus, to detect this we need the model to find relations like, "token X at t=0,i=83 is similar to token Y at t=1,i=120, and token Y at t=0,i=39 is similar to token X at t=1,i=125." We want it to be able to recognize this as a vibrato, and then see what that vibrato entails: is it a natural vibrato? Is it a momentary flourish or the main attraction at the time?

**Vocal Fry / Creak:** This can appear as a strong sub-harmonics in the pitch distribution. The model needs to detect the pattern of "strong activation at bin X and also at bin X/2." This is a global pattern that localized pattern finding models like convolution models might miss entirely without making the model extensively deep with multiple kernel sizes, and we don't want that. We hope that the model can see this, and recognize what *sort* of vocal fry it is. Is it that metal sort of fry that sounds like the abyss? Is it that sort of unintentional creak that happens when singing an emotional song?

**Shouts / Strain:** This will appear as a combination of a high fundamental frequency and a change in the energy distribution across the harmonics (spectral tilt). We need to capture both. The high F0 will be somewhere up top, and the change in timbre will be encoded in the relative values of those sections of the CREPE embedding across the two timesteps. We hope that the model will be able to identity what those patterns entail: are these regular shouts across the room? Are these intentional and consistent over both timesteps? Is it "controlled" somehow? If so, how is it controlled?

**Whispers / Breathy Voice:** This will appear as a very "flat" or "noisy" CREPE embedding with low confidence. The model needs to recognize the lack of discernable information (or hidden information, if there are any, as we hope for when picking the tiny model over the full CREPE model. We'll talk about this later.) and map it to a specific kind of conditioning vector that tells the CFM to generate a noisy, un-pitched mel-spectrogram, but to generate it in a specific way in matching with the source. Was the source trying to do a low unpitched whisper with their voice like they were speaking into someone's ears? Or is this a ghostly whisper for acting out a scene?

We immediately see that pretty much all of these require *global* patterns, and the model will need to look at various spots that might seem random to find complex, human-defined meanings.

If at this point, your mind is creaming "ATTENTION BLOCKS!!!" at you, then yes, you are absolutely correct.

If the model can infer all this, it will in essence become a pitch contour understanding module, taking the two raw CREPE frames in and inferring exactly what the performance details are to replicate it if it had to happen with a different set of vocal cords, which is *exactly* what we wanted.

How we apply it is slightly nuanced. Attention works across tokens, and in this case we don't have a very obvious way to tokenize the CREPE embeddings.
Here are a few approaches:
1. Timestep axis as sequence, 256 dimensional CREPE embedding as tokens.
This is bad. First off, having just two tokens means that in the end, our entire attention block becomes a very sophisticated model designed to calculate the "a" and "b" in aX + bY, twice. That is horribly inefficient and handicaps the model greatly.
It is also bad since we cannot get the model to learn the relationship *between different pitches*. That is an essential thing to have.
2. Timestep axis as tokens, 256 dimensional CREPE embedding as sequence.
This is better, but there are a couple of issues with it. Having 256 sequence lengths immediately makes the model all the more computationally expensive, and I'd like this model to not require more than 6gb of VRAM to run so that we can all use it at home on laptops. There's also perhaps the more important concern regarding the model's capabilities itself of the model being overkill; if it is too big, we risk the model beginning to learn things we didn't want it to learn. We should keep the model at a size where it is forced to do only what it needs to, and this would be approaching the upper limit of that.
The final other reason this is bad is because now we don't have a way to learn relationships between different *timesteps*. We solve one problem, and we introduce another.
3. Each individual CREPE embedding value as a token (like: (crepe_t0_i0, crepe_t0_i1, ..., crepe_t1_i0, ...)).
Theoretically the best, but also computationally the heaviest, and has an even bigger problem of potentially learning too *much*.

We don't do any of this and try to strike a good balance instead.

4. The approach we use. Put the CREPE embedding values into "super-bins" of size `coarse_factor`. For example, with the model's `coarse_factor` of 8, we will have tokens of the form (crepe_t0_i0, ..., crepe_t0_i7), and so on. We will then have 32 of these tokens in each timestep, and 64 of them in total. This is a wonderful balance. The attention mechanism still gets, all things considered, a very fine-grained view of the pitch information within the embeddings, with 64 tokens to play around with, and this is across both "frequency" (more specifically CREPE embedding values) and time.

This is exactly what we want! It's doing what LLMs do with seemingly meaningless words and tokens, and doing it with pitch, trying to understand the human meaning behind those changes to facilitate with recreating it completely from scratch even with every single pitch being completely different from the original's. The intent is fully preserved, and the decoder model can do it without breaking a *sweat*, thanks to our new encoder model's contextualized embeddings.

With this in mind, the architecture.

### Picking the right CREPE model

This is the first step, the input to our attention model.

While researching CREPE, I came upon this very interesting article: https://medium.com/axinc-ai/crepe-a-machine-learning-model-for-high-precision-pitch-estimation-8562d83d44a5

The mention of the difference between the tiny and the full CREPE models is especially interesting. The author notes that, apparently, the full model successfully outputs a very low confidence score in unvoiced sections, while the tiny model fluctuates during those same unvoiced sections.
However, crucially, **for voiced sections it does pretty well**.

Immediately then, using the tiny model has an immense benefit.

Before settling on this final architecture, I'd tried various other architectures, and one of the things I found was that with those earlier attempts, there was a massive problem where because the CREPE information was so *good*, the model tended to end up getting lazy and relying heavily on it, taking shortcuts (this is before the attention mechanism, although if applied now, even with the attention mechanism it is likely the model will just learn to turn the attention mechanism into an identity function, or something close to it in utility).

S3 tokens do not care about pitch and only about prosody and the contents of someone's speech, right? This is how the original model works. Thus, the decoder model actually had a very difficult job in its original form. It did not have any information on the pitch of the person's speech, only the contents, so it basically had to just guess at what the right pitch was for each frame based on the way they spoke the words and their timbre x-vector.

While this crippled the model's ability to do any singing or fine pitch control, it *also* gave it the amazing ability to create speech patterns that were incredibly lively, with growls, shouts, grunts, and more. But once I provided it with the CREPE pitch information, suddenly all of that becomes almost unnecessary. It could get pretty good losses just by outputting something in that general pitch range. This is terrible, catastrophic forgetting.

So immediately, by using the tiny model, we get some intense scrambling of the pitch embeddings during unvoiced sections, forcing the model to adapt in those cases. We add on top of this a fixed dropout of having 20% of each batch having their pitch understanding conditioning vectors zeroed out, meaning the model can no longer focus entirely on the pitch vectors and forget about how to deal with those more special vocal sounds that we can produce.

There's a further layer to this, however.

Sometimes, larger models can learn something so well that it learns well to ignore anything and everything that is unrelated to its task.

The full model, being highly accurate and stable, might have learned well what constitutes a "voiced" sound and what doesn't. When it encounters an unvoiced segment (like an 's' sound or a breath) it could perhaps then get rid of that "noise" early on.
In contrast, the tiny model is less accurate and more "excitable."
During unvoiced segments, instead of confidently outputting zero, it can "hallucinate" a rapidly fluctuating, noisy F0 contour, and its embeddings might be messy. However, even though its outputs are "messy", this is actually a good sign that it has not yet learned how exactly to filter those sections out, ignore them, and push them to zero.
It is thus possible that the tiny model's "fluctuations" actually contain useful information about unvoiced sections!
The "hallucinations" of the "tiny" model during unvoiced parts might not just be noise; they might be a unique, information-rich fingerprint of the unvoiced sound that our encoder model can figure out the intent and meaning of to pass along to the decoder.
The "full" model would just output "zero" for all of these events. It throws away this information in favor of being "correct" about the absence of a fundamental frequency. The "tiny" model, in its uncertainty, might accidentally preserve a rich signature of the unvoiced texture.
If this is true, then our transformer model can now also become an additional point of reference for unpitched tones as well, in addition to serving as the definitive reference for pitch information, creating a mutually-beneficial relationship with the s3 tokens.

This is good. Thus, we pick the tiny model.

### The Main Model Architecture

The sizes are tuned carefully.

16 heads correspond somewhat to the number of common vocal techniques.
Vibrato, growls, grunts, puffs, ... You can imagine that there'd be around 16 of these in the most general "category" sense of distinct vocal techniques that have different characteristics.
Of course, we always hope that the model can make better use of the 16 heads than what we can plan for them.

As mentioned earlier, 8 CREPE embedding values per bin should provide enough granularity to allow for the transformer to do its job well too.

Keys and values are of 8 dimensions, which means that the key and value projection linear models will be a linear transformation from 8 dimensions to 8 dimensions. That should be a pretty informative transformation; not creating information out of thin air, but also giving the model as much room as we can give it to look up information, giving it the full 8 token dimensions, just processed.

For the standard feed forward network that follows, we do 2x instead of the usual 4x `d_ff` for its hidden layer. This gives the model an extra chance to adjust its learned attention-attended features, and is standard.

Finally, we have the projection layer which goes from 1536 dimensions to the 80 dimensions for conditioning. It has the absolute most amount of information by far to work with, having the entire transformer network's output to understand and interpret. We hope that this can pick up some of the slack of more fine-grained relationship learning, as this is the last chance for the model to do so before the conditioning vector is passed along. It has the monumental task of summarizing all the complex harmonic and temporal patterns discovered by the transformer into the 80-dimensional vector that the `CausalConditionalCFM` needs.

### Positional Encoding

Transformers need positional encodings, so here we apply them with a simple linspace from [-1, 1].
For each single value in the original 256-dimensional CREPE embedding vector, we turn that into three values, (`crepe_tX_iY`, `Y` mapped to [-1, 1], `X` mapped to [-1, 1]).

This deviates from the usual approach of learned positional encodings, but for good reason. Text models have positions that are "soft", as in, you could easily swap sentences around to say the same thing in a variety of different ways with the same tokens in a variety of positions. This is not the case with our tokens, if we say a token is at 50hz (conceptual, since CREPE embedding bins probably don't correspond to pitch this directly) and in timestep 0, we mean the token is at 50hz and in timestep 0. Giving the model room to interpret here would actually be *handicapping* the model by giving it the chance to swap timesteps around and swap frequencies around, which we don't need. And as much as we want the model to learn intent, we don't want it to learn it at such a level of "this is a vibrato"; we want it to learn as much as it can to facilitate the recreation of said vocal "act" with a different vocal cord. So, simple linspace it is.


And that's the model!

Through consideration after consideration, we have slowly arrived at a model architecture that should be incredibly capable at providing the rich pitch-related performance context that our decoder is sorely lacking right now in order to be able to handle much more complicated vocal sounds.

We have finally conquered the first great hurdle of Machine Learning.


However, before we can finally put this design to the test, the elephant in the room makes the atmosphere feel heavy. We must now tackle the second great hurdle of Machine Learning, and perhaps where most attempts at building a model like this stops.

### Data

If you've ever tried to find a singing dataset... you'll know it's a desert out there.

While there are dozens of amazing high-quality emotion-labelled speech datasets provided by countless amazing prior researchers available online, there's barely *any* acapella datasets available... Which is especially sad for me, because I love singing! And wouldn't it be just great to apply my math love to this other love and have them marry each other? A timbre changing model was always on my radar as a really cool thing to be able to do, but alas, it didn't seem very plausible to train a model like that when you didn't have the sort of million buck licensing agreements that firms can get.

But I've also always believed in another thing, which is that constraints can possibly beget great creativity. This is because in constraits, you see that the problem might not have been as intractable as you might have initially thought... that because the constraits force you to always be on your toes about what you are able to do, you can find seemingly obvious straightforward and logical paths towards your goal, that you had for some reason missed up until now.

Before starting anything, it's time to bite the bullet and spend a couple days finding as many vocal datasets as we can get our hands on. First, datasets related to Speech Emotion Recognition. These datasets often contain very emotive speech samples that span a much wider range of possible vocal sounds, but only for speaking tasks. Here is the result of that gruelling process. There are more out there, and we can't be picky about what to include, but some had major issues like samples that were extremely noisy, so those were left out.

- *CAMEO, a collection of various SER (Speech Emotion Recognition) learning related datasets, which includes:*
- CREMA-D (7442 clips)
- CaFE (936 clips)
- Emozionalmente (6902 clips)
- JL-Corpus (2400 clips)
- Mexican Emotional Speech Database (MESD) (862 clips)
- nEMO (4481 clips)
- Oreau (502 clips)
- RAVDESS (1440 clips)
- RESD (1396 clips)
- *And then,*
- AESDD (604 clips)
- EmoDB (535 clips)
- CASIA (600 clips)
- SAVEE (480 clips)
- Emov_DB (6893 clips)
- ESD (35000 clips (!!))
- BERSt (4523 clips)
- SHEMO (3000 clips)
- emoUERJ (285 clips)
- JVNV (1615 clips)
- YueMotion (1080 clips)

You'll immediately recognize that these have languages all *over* the place.

This is intentional. The key realization is that talking *is* singing, and vice versa! In many languages, emotional speech often ends up sounding sing-song, with a clear discernable pitch pattern and pitch control.

To get an idea of this, just go over to the JVNV dataset and listen to a couple of samples. *These are incredibly valuable for us*. If we can't get singing, then hell we're gonna get as much talking that sounds like singing as possible.

And here comes the most important realization of all.

Hold on... **all these languages present entirely different pitch patterns, different way to pronounce things, different contours, and I *bet* that taken *together* all at once, we could perhaps learn a pretty damn good representation of all the different ways a person could utilize their voices!**

How do we facilitate that? The Law of Large Numbers swoops in to save the day once again.

Remember how we were talking about CFM? Each suggestion of a path from the noise region to the real data region might be suboptimal and bad, but *averaged out*, we end up with a smooth, coherent, sensible direction towards it!

**We hope that we can do the exact same thing here.**

To reinforce this hope, we must structure our training appropriately. Gone are the usual random batching, that won't do. We have too little data to just chuck an RNG on the dataset and call it a day. We need to be intentful at each training step, giving the model only one way to make itself better, which is to learn the most universal representations of human speech itentions and their associated patterns.

To do this, each batch must be controlled carefully so that there is never a situation where the model gets a batch of size 32 with 31 of the samples being from CREMA-D, for example, allowing it to get a lower loss by just specializing in English pronounciations. No. That would completely destroy our model, but it is very likely to happen if we just allowed things to progress randomly.

Instead, each batch will have an *exact* number of samples taken from each dataset.

Actually, we don't really have enough large datasets to do this, do we?

**So we have to pool them together.**

What we do is that, since each dataset might not be large enough to sustain one sample from each on every batch, we group models with characteristics we hope to learn from them, and then pull samples from *those* instead.

The actual pooling requires a lot of considerations, but you can see what I ended up with in the `bake/src/config.rs` file within the Github repository.

Ah, right, I haven't yet listed the musical datasets I'd found, right? Don't worry, it'll be quick...

- VocalSet (3613 clips | 10.1 hours of monophonic recorded audio of professional singers, from 20 different singers (9 male, 11 female) and a range of voice types!!! There are no actual songs as far as I can tell, but instead just things like scale runs, sound effects, and so on that you might find a choir class doing for warmup before the start of class)
- acapella (132 clips | Mandarin acapella singing)
- bel_canto (203 clips | Traditional Mandarin Opera Singing)

Yeah, it's not a lot. So what we *do* have, we need to make the absolute most out of!!

Some groupings are done based on language ancestry and characteristics, while others are for training reasons. We give 8 slots that are guaranteed for the `VocalSet` in each batch because it is so hard to come by that we really don't have another choice.

And once we have batches constructed in this way, the hope is simple: That *all* these distinct ways of producing vocal sounds that try to pull the model in *their* direction will average out at *each* step to the most comprehensive, most generalizable universal pitch-based intent encoding that we can get. This is *all* with less than 1% of the training dataset being actual singing... Is this going to work?

It sounds incredibly promising, but now we need to put it into action.

This is where hell begins.

### Optimizations

My initial attempts were plagued by rented GPUs on a clock running at around 0% and `load_dataset` being completely stuck. The special batching mechanism that is a hard requirement for making this model work was done with `interleave_datasets` at first but it seemed to just freeze there. Different datasets downloaded differently and I hit rate-limits trying to download BERSt when loading it which was stored as a couple hundred thousand small audio files. And oh the absolutely joy of finding out that the reason it was stuck there was because... well, I still don't quite know, really. I am sure that downloading had something to do with it, perhaps the library doesn't like it when you simultaneously try to download from 23 datasets. Or perhaps it is the buffering or processing parts. They were set to streaming and everything, but in the end, I just wrote it down to "people don't usually train like this". I mean, to be fair, you'd be forgiven to think, "oh, why not just set the probabilities to *around* what you want the batch composition to be and have it average out?" But that is a compromise, not a benefit; if our theory about the model is correct, not only are we giving up the chance to train it *far* more efficiently, we are also deliberately crippling the model by tempting it into specializing.

But obviously that wasn't doing it, so after more of desperately trying to make it work with the usual approaches, I decided to go full-blown guns-blazing custom software.

Firstly, you'll notice a directory called `bake` inside of this repository. *That*, is a highly efficient, *highly* parallelized Rustlang binary that takes a massive number of datasets and combines them in the exact way we discussed, builds said dataset with sample rate conversions already done (another suspect for the slowness before, by the way), and then forms nice parquet chunks formatted with a single batch per row, and with the audio aligned perfectly with the parquet audio file column standards. Now, the training machine is relieved of:

- Collecting 23 datasets
- Randomizing said datasets and putting them into groups
- Then pulling from those groups ensuring that not one sample is missed to form batches for each training step
- And then resampling those myriad of audio samples into the 24000hz the model needs it to be.

All of this is done already, and in fact, you can get this whole pre-batching and preparation of data done using that binary in **20 minutes** on a high-specced rented server. I was filled with glee when I came back from getting water expecting it to be still running, only to find it giving me a green checkmark and a "Bake complete! Output written to directory-"

Once you upload the dataset onto a remote repository like Huggingface for example, the training script takes over, and it is as simple and as efficient as it can be. Just prefetch a couple of parquet slices in a rolling fashion (a custom streaming dataset loader compatible with torch is provided, since I found it to be much more reliable and transparent given the very specific requirements of our training approach, and critically to avoid a strange error that occurs within pyarrow with datasets containing massive rows that doesn't occur with arrow-rs), and each step is just reading a single row; the batch is already there.

We also grab only a 4s random segment from each sample for each step; since many of the samples are far longer than 4s, this means that each epoch is seeing new data and effectively running on a new sample, even if some of its contents are overlapping.

With all this work, we have successfully stretched the little data we have on human speech and multiplied its power many fold by being careful about how to do the training.

### The Moment of Truth

The training script is inside of `src/train/train.py`, and the dataset has already been built and uploaded here `https://huggingface.co/datasets/Bill13579/combined-set-1`.

*deep breath*.

We start the script, and we see that it has reached 999 batches. Remember that YouTube video? Now, listen starting from `8:00`. 999 batches isn't much at all, but we should check if the training script is not bugged.

...

Are you back?

Then, HOLY F\*CK. AT 999 STEPS, THE PITCH IS ALREADY *EXACTLY* MATCHED!!!

Now let us listen from `12:00` until, at last, the end of the video.

This is the result we obtain at 117580 steps into training of this final version of the model, after 3 days of training on a single rented 3090 using our optimized training pattern, after around 10 epochs in total through the entire dataset.

I'd be lying if I said I didn't have tears building in my eyes when I finally heard it, giggling like a maniac at the results; maybe it did that for some of you too. Let me know if it did so that I can repeat the same thing for *that* this time.

