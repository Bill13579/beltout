---
license: mit
---

***For everyone returning to this repository for a second time, I've updated the Tips and Examples section with important information on usage, as well as another example. Please take a look at them for me! They are marked in square brackets with [EDIT] and [NEW] so that you can quickly pinpoint and read the new parts.***

# BeltOut

They say timbre is the only thing you can't change about your voice... well, not anymore.

**BeltOut** is **the world's first *pitch-perfect*, zero-shot, voice-to-voice timbre transfer model with *a generalized understanding of timbre and how it affects delivery of performances*.** It is based on ChatterboxVC.

**[NEW] To first give an overhead view of what this model does:**

First, it is important to establish a key idea about why your voice sounds the way it does. There are two parts to voice, the part you *can* control, and the part you *can't*.

For example, I can play around with my voice. I can make it sound *deeper*, more resonant by speaking from my chest, make it sound boomy and *lower*. I can also make the pitch go a lot higher and tighten my throat to make it sound sharper, more *piercing* like a cartoon character. With training, you can do a lot with your voice.

What you cannot do, no matter what, though, is change your ***timbre***. **Timbre** is the reason why different musical instruments playing the same note sounds different, and you can tell if it's coming from a violin or a flute or a saxophone. It is *also* why we can identify each other's voices.

It can't be changed because it is *dictated by your head shape, throat shape, shape of your nose, etc.* With a bunch of training you can alter a lot of qualities about your voice, but someone with a mid-heavy face might always be louder and have a distinct "shouty" quality to their voice, while others might always have a rumbling low tone.

The model's job, and its *only* job, is to change *this* part. *Everything else is left to the original performance.* This is different from most models you might have come across before, where the model is allowed to freely change everything about an original performance, subtly adding an intonation here, subtly increasing the sharpness of a word there, to fit the timbre. This model does not do that, disciplining itself to strictly change only the timbre part.

So the way the model operates, is that it takes 192 numbers representing a unique voice/timbre, and also a random voice recording, and produces a new voice recording with that timbre applied, and *only* that timbre applied, leaving the rest of the performance entirely to the user.

**Now for the original, slightly more technical explanation of the model:**

It is explicitly different from existing voice-to-voice Voice Cloning models, in the way that it is not just entirely unconcerned with modifying anything other than timbre, but is even more importantly *entirely unconcerned with the specific timbre to map into* while still maintaining that separation of timbre and all other controllable performance details. The goal of the model is to learn how differences in vocal cords and head shape and all of those factors that contribute to the immutable timbre of a voice affects delivery of vocal intent *in general*, so that it can guess how the same performance will sound out of such a different base physical timbre.

This model represents timbre as just a list of 192 numbers, the **x-vector**. Taking this in along with your audio recording, the model creates a new recording, guessing how the same vocal sounds and intended effect would have sounded coming out of a different vocal cord.

In essence, instead of the usual `Performance -> Timbre Stripper -> Timbre "Painter" for a Specific Cloned Voice`, the model is a timbre shifter. It does `Performance -> Universal Timbre Shifter -> Performance with Desired Timbre`.

This allows for unprecedented control in singing, because as they say, timbre is the only thing you truly cannot hope to change without literally changing how your head is shaped; everything else can be controlled by you with practice, and this model gives you the freedom to do so while also giving you a way to change that last, immutable part.

**Now go belt your heart out~**

# Some Points
- Small, running comfortably on my 6gb laptop 3060
- *Extremely* expressive emotional preservation, translating feel across timbres
- Preserves singing details like precise fine-grained vibrato, shouting notes, intonation with ease
- Adapts the original audio signal's timbre-reliant performance details, such as the ability to hit higher notes, very well to otherwise difficult timbres where such things are harder
- Incredibly powerful, doing all of this with just a single x-vector and the source audio file. No need for any reference audio files; in fact you can just generate a random 192 dimensional vector and it will generate a result that sounds like a completely new timbre
- Architecturally, only 335 out of all training samples in the 84,924 audio files large dataset was actually "singing with words", with an additional 3500 or so being scale runs from the VocalSet dataset. Singing with words is emergent and entirely learned by the model itself, learning singing despite mostly seeing SER data
- Open-source like all my software has been for the past decade.
- Make sure to read the [technical report](./TECHNICAL_REPORT.md)!! Trust me, it's a fun ride with twists and turns, ups and downs, and so much more.

Join the Discord [https://discord.gg/MJzxacYQ](https://discord.gg/MJzxacYQ)!!!!! It's less about anything and more about I wanna hear what amazing things you do with it.

# Examples and Tips

The x-vectors, and the source audio recordings are both available on the repositories under the `examples` folder for reproduction.

**[EDIT] Important note on generating x-vectors from sample target speaker voice recordings: Make sure to get as much as possible.** It is highly recommended you let the analyzer take a look at at least 2 minutes of the target speaker's voice. More can be incredibly helpful. If analyzing the entire file at once is not possible, you might need to let the analyzer operate in chunks and then average the vector out. In such a case, after dragging the audio file in, wait for the `Chunk Size (s)` slider to appear beneath the `Weight` slider, and then set it to a value other than `0`. A value of around 40 to 50 seconds works great in my experience.

`sd-01*.wav` on the repo, [https://youtu.be/5EwvLR8XOts](https://youtu.be/5EwvLR8XOts) (output) / [https://youtu.be/wNTfxwtg3pU](https://youtu.be/wNTfxwtg3pU) (input, yours truly)

`sd-02*.wav` on the repo, [https://youtu.be/KodmJ2HkWeg](https://youtu.be/KodmJ2HkWeg) (output) / [https://youtu.be/H9xkWPKtVN0](https://youtu.be/H9xkWPKtVN0) (input)

**[NEW]** [https://youtu.be/E4r2vdrCXME](https://youtu.be/E4r2vdrCXME) (output) / [https://youtu.be/9mmmFv7H8AU](https://youtu.be/9mmmFv7H8AU) (input) (Note that although the input *sounds* like it was recorded willy-nilly, this input is actually after **more than a dozen takes**. The input is not random, if you listen closely you'll realize that if you do not look at the timbre, the rhythm, the pitch contour, and the intonations are all carefully controlled. The laid back nature of the source recording is intentional as well. Thus, only because *everything other than timbre is managed carefully*, when the model applies the timbre on top, it can sound realistic.)

Note that a very important thing to know about this model is that it is a *vocal timbre* transfer model. The details on how this is the case is inside the technical reports, but the result is that, unlike voice-to-voice models that try to help you out by fixing performance details that might be hard to do in the target timbre, and thus simultaneously either destroy certain parts of the original performance or make it "better", so to say, but removing control from you, this model will not do any of the heavy-lifting of making the performance match that timbre for you!! In fact, it was **actively designed to restrain itself from doing so, since the model might otherwise find that changing performance details is the easier to way move towards its learning objective.**

So you'll need to do that part.

Thus, when recording with the purpose of converting with the model later, you'll need to be mindful and perform accordingly. For example, listen to this clip of a recording I did of Falco Lombardi from `0:00` to `0:30`: [https://youtu.be/o5pu7fjr9Rs](https://youtu.be/o5pu7fjr9Rs)

Pause at `0:30`. This performance would be adequate for many characters, but for this specific timbre, the result is unsatisfying. Listen from `0:30` to `1:00` to hear the result.

To fix this, the performance has to change accordingly. Listen from `1:00` to `1:30` for the new performance, also from yours truly ('s completely dead throat after around 50 takes).

Then, listen to the result from `1:30` to `2:00`. It is a marked improvement.

Sometimes however, with certain timbres like Falco here, the model still doesn't get it exactly right. I've decided to include such an example instead of sweeping it under the rug. In this case, I've found that a trick can be utilized to help the model sort of "exaggerate" its application of the x-vector in order to have it more confidently apply the new timbre and its learned nuances. It is very simple: we simply make the magnitude of the x-vector bigger. In this case by 2 times. You can imagine that doubling it will cause the network to essentially double whatever processing it used to do, thereby making deeper changes. There is a small drop in fidelity, but the increase in the final performance is well worth it. Listen from `2:00` to `2:30`.

**[EDIT] You can do this trick in the Gradio interface. Simply set the `Weight` slider to beyond 1.0. In my experience, values up to 2.5 can be interesting for certain voice vectors. In fact, for some voices this is necessary!** For example, the third example of Johnny Silverhand from above has a weight of 1.7 applied to it (the `npy` file in the repository already has this weighting factor baked into it, so if you are recreating the example output, you should keep the weight at 1.0, but it is important to keep this in mind while creating your own x-vectors).

Another tip is that in the Gradio interface, you can calculate a statistical average of the x-vectors of massive sample audio files; make sure to utilize it, and play around with the Chunk Size as well. I've found that the larger the chunk you can fit into VRAM, the better the resulting vectors, so a chunk size of 40s sounds better than 10s for me; however, this is subjective and your mileage may vary. Trust your ears.

# Installation

Installation is simple, but should be done in a clean anaconda environment.
```shell
# conda create -n beltout python=3.12
# conda activate beltout

# IMPORTANT: Tell Git to not download any checkpoints by setting GIT_LFS_SKIP_SMUDGE to 1! The commands for that are down below and different on each platform.

# PowerShell (Windows)
$Env:GIT_LFS_SKIP_SMUDGE="1"
git clone https://github.com/Bill13579/beltout.git

# CMD (Windows)
set GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/Bill13579/beltout.git

# Unix
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Bill13579/beltout.git

cd beltout
pip install -e .
```
Versions of dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode, unlike with the pip installation.

Once that is done, go to Huggingface to download a checkpoint for each model and place it inside `checkpoints`. After which you can do:

```shell
python run.py
```

To start the Gradio interface.

# Supported Lanugage
The model was trained on a variety of languages, and not just speech. Shouts, belting, rasping, head voice, ...

As a baseline, I have tested Japanese, and it worked pretty well.

In general, the aim with this model was to get it to learn how different sounds created by human voices would've sounded produced out of a different physical vocal cord. This was done using various techniques while training, detailed in the technical sections. Thus, the supported types of vocalizations is vastly higher than TTS models or even other voice-to-voice models.

However, since the model's job is *only* to make sure your voice has a new timbre, the result will only sound natural if you give a performance matching (or compatible in some way) with that timbre. For example, asking the model to apply a low, deep timbre to a soprano opera voice recording will probably result in something bad.

Try it out, let me know how it handles what you throw at it!

# Acknowledgements
- [Chatterbox](https://github.com/resemble-ai/chatterbox)
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Technical Report

It's quite the something and ended up being very long, but it's a great read I think. See it [here](./TECHNICAL_REPORT.md)!!

# Socials

There's a [Discord](https://discord.gg/MJzxacYQ) where people gather; hop on, share your singing or voice acting or machine learning or anything! It might not be exactly what you expect, but I have a feeling you'll like it. ;)

My personal socials: [Github](https://github.com/Bill13579), [Huggingface](https://huggingface.co/Bill13579), [LinkedIn](https://www.linkedin.com/in/shiko-kudo-a44b86339/), [BlueSky](https://bsky.app/profile/kudoshiko.bsky.social), [X/Twitter](https://x.com/kudoshiko), []()

# Closing

This ain't the closing, you kidding!?? I'm so incredibly excited to finally get this out I'm going to be around for days weeks months hearing people experience the joy of getting to suddenly play around with a infinite amount of new timbres from the one they had up, and hearing their performances. I know I felt that way...

Yes, I'm sure that a new model will come soon to displace all this, but, speaking of which...

# Call to train

If you read through the technical report, you might be surprised to learn among other things just how incredibly quickly this model was trained.

It wasn't without difficulties; each problem solved in that report was days spent gruelling over a solution. However, I was surprised myself even that in the end, with the right considerations, optimizations, and head-strong persistence, many many problems ended up with extremely elegant solutions that would have frankly never come up without the restrictions.

And this just proves more that people doing training locally isn't just feasible, isn't just interesting and fun (although that's what I'd argue is the most important part to never lose sight of), but incredibly important.

So please, train a model, share it with all of us. Share it on as many places as you possibly can so that it will be there always. This is how local AI goes round, right? I'll be waiting, always, and hungry for more.

\- Shiko

