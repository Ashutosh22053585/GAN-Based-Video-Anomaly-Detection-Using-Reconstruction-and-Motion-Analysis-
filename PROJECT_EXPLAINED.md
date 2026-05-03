# PROJECT_EXPLAINED.md

## 1. The Big Picture
Imagine you are hiring a security guard to watch a quiet, boring hallway. Instead of showing the guard thousands of videos of robberies, fights, and fires (which are rare and hard to collect), you just let the guard watch the empty hallway for 100 hours. The guard learns exactly what "normal" looks like—how the shadows move, how the doors open, the regular pace of people walking. Because the guard knows "normal" so perfectly, the second someone runs down the hall or a fight breaks out, the guard instantly knows something is wrong because it doesn't fit the pattern. Our AI is exactly like that guard: we train it only on normal videos to perfectly reconstruct normal scenes, and when it drastically fails to reconstruct a scene, it raises an alarm!

---

## 2. Project Folder Structure

*   **`config.py`** — "The Rulebook": Holds all the master settings, folder paths, and numbers (like how fast the AI learns) so every other file knows what the rules are.
*   **`dataloader.py`** — "The Librarian": Fetches the video frames from your hard drive, pairs them up, calculates how things are moving between them, and hands them to the AI in neat stacks.
*   **`generator.py`** — "The Artist": Tries to draw/reconstruct a perfect video frame based on the previous frame and the movement it sees.
*   **`discriminator.py`** — "The Art Critic": Looks at the Artist's drawing and tries to figure out if it's a real photograph or a fake drawing.
*   **`loss.py`** — "The Grader": Calculates the exact score of how badly the Artist or the Critic messed up so they can learn from their mistakes.
*   **`train.py`** — "The Gym": The main loop where the Artist and the Critic practice against each other millions of times to get better.
*   **`evaluate.py`** — "The Final Exam": Takes the fully trained Artist, puts it in the real world to watch new videos, and uses its mistakes to flag security anomalies.

---

## 3. File by File Explanation

### `config.py` — "The Rulebook"
**What problem does this file solve?**
It keeps all the hardcoded numbers (like image sizes and learning speeds) in one place so you don't have to hunt through the code to change a setting.

**The analogy:** 
It’s like the settings menu in a video game. You change the volume or resolution here, and the whole game updates automatically.

**How it actually works:**
```python
# We define how big the images should be when we feed them to the AI
IMG_HEIGHT  = 256       # Resize the image height to exactly 256 pixels
IMG_WIDTH   = 256       # Resize the image width to exactly 256 pixels

# The number of images the AI looks at simultaneously before updating its brain
BATCH_SIZE  = 16        # The AI will process 16 images at a time (a stack of 16)

# The learning rate defines how big of a step the AI takes when correcting mistakes
LR_G        = 2e-4      # The Artist (Generator) changes its brain by 0.0002 each mistake
```
*Why we made this choice:* Centralizing settings prevents bugs where you change an image size in one file but forget to change it in another. If you lower `BATCH_SIZE` to 8, the AI takes up less computer memory but trains slower. If you raise `LR_G` to `0.1`, the AI will try to learn too fast and will likely become unstable and fail.

---

### `dataloader.py` — "The Librarian"
**What problem does this file solve?**
Neural networks can't read video files directly. They need raw numbers. This file reads the images, calculates movement, and bundles them into mathematical tensors (stacks of numbers).

**The analogy:**
Like a sous-chef who washes, chops, and portions all the ingredients so the head chef (the AI) can just focus on cooking.

**How it actually works:**
```python
# We are preparing one specific training example for the AI to look at
def __getitem__(self, idx: int) -> dict: # Function that gets one specific item from the dataset
    prev_path, curr_path = self.samples[idx] # Grab the file paths for the previous frame and current frame
    
    prev_gray_u8 = self._load_gray_uint8(prev_path) # Load the previous frame as a black-and-white image
    curr_gray_u8 = self._load_gray_uint8(curr_path) # Load the current frame as a black-and-white image
    
    prev_resized = cv2.resize(prev_gray_u8, (256, 256)) # Shrink previous frame to 256x256 pixels
    curr_resized = cv2.resize(curr_gray_u8, (256, 256)) # Shrink current frame to 256x256 pixels
    
    flow = compute_optical_flow(prev_resized, curr_resized) # Calculate where pixels moved between the two frames
    flow = normalize_flow(flow) # Shrink the movement numbers to be strictly between -1.0 and 1.0
    
    curr_norm = curr_resized.astype(np.float32) / 127.5 - 1.0 # Convert image colors (0-255) to decimals between -1.0 and 1.0
    
    frame_tensor = torch.from_numpy(curr_norm).unsqueeze(0) # Convert the image into a PyTorch math object (tensor)
    flow_tensor  = torch.from_numpy(flow).permute(2, 0, 1) # Rearrange the movement data into a PyTorch math object
    input_tensor = torch.cat([frame_tensor, flow_tensor], dim=0) # Glue the image and the movement data together like a sandwich
    
    return {"input": input_tensor, "target": frame_tensor} # Hand this sandwich to the AI to practice on
```
*Why we made this choice:* We resize to 256x256 because neural networks need uniform shapes, and 256 is a power of 2, which divides cleanly. We scale colors to `[-1, 1]` because neural networks learn math much faster when numbers are small and centered around zero.

---

### `generator.py` — "The Artist"
**What problem does this file solve?**
It is the core brain of the system. It takes the previous frame and the movement data, and tries to draw what the current frame should look like.

**The analogy:**
Like an artist who looks at a sketch through a blurry magnifying glass (Encoder), gets the general vibe, and then painstakingly draws a highly detailed painting on a new canvas (Decoder).

**How it actually works:**
```python
# This is the forward pass, meaning "take the input and push it through the brain"
def forward(self, x: torch.Tensor) -> torch.Tensor: # Define how data flows through the Artist
    e1 = self.enc1(x) # Encoder step 1: Look at the image and extract basic edges
    e2 = self.enc2(e1) # Encoder step 2: Shrink the image and look for textures
    e3 = self.enc3(e2) # Encoder step 3: Shrink further and look for objects
    e4 = self.enc4(e3) # Encoder step 4: Shrink to a tiny grid to understand the whole scene
    
    b  = self.bottleneck(e4) # The Bottleneck: Compress everything into raw concepts, then expand slightly
    
    d4 = self.dec4(torch.cat([b,  e4], dim=1)) # Decoder 1: Expand and glue on details from e4 (Skip Connection)
    d3 = self.dec3(torch.cat([d4, e3], dim=1)) # Decoder 2: Expand and glue on details from e3 (Skip Connection)
    d2 = self.dec2(torch.cat([d3, e2], dim=1)) # Decoder 3: Expand and glue on details from e2 (Skip Connection)
    
    out = self.output_layer(torch.cat([d2, e1], dim=1)) # Final Polish: Expand to full size, glue on e1 details, output picture
    return out # Hand back the finished drawing
```
*Why we made this choice:* We chose a **U-Net**. A U-Net shrinks the image to understand the "big picture" (like knowing there's a person walking), but shrinks destroy fine details (like the person's shoelaces). "Skip connections" (`torch.cat`) literally copy-paste the high-res details from the early steps directly to the end steps so the final drawing is crystal clear.

---

### `train.py` — "The Gym"
**What problem does this file solve?**
It orchestrates the learning process, letting the Generator (Artist) and Discriminator (Critic) take turns updating their brains.

**The analogy:**
Like a sports coach blowing a whistle: "Critic, judge this fake art! Now update your rules. Okay, Artist, you try to fool the critic! Now update your skills. Repeat 10,000 times."

**How it actually works:**
```python
# This happens once for every stack of images
def train_step(input_t, prev_t, target_t, netG, netD, optG, optD, criterion): # The function to run one training drill
    generated = netG(input_t) # The Artist creates a fake frame based on the inputs
    
    optD.zero_grad() # Erase the Critic's old memories/math from the previous drill
    pred_real = netD(input_t, target_t) # The Critic looks at the REAL video frame and gives a score
    pred_fake = netD(input_t, generated.detach()) # The Critic looks at the FAKE frame (we detach it so we don't accidentally update the Artist yet)
    loss_D = criterion.discriminator_total(pred_real, pred_fake) # The Grader calculates how badly the Critic got tricked
    loss_D.backward() # Calculate the exact math adjustments needed to make the Critic smarter
    optD.step() # Apply the adjustments to the Critic's brain
    
    optG.zero_grad() # Erase the Artist's old memories/math from the previous drill
    pred_fake_for_G = netD(input_t, generated) # The Critic looks at the FAKE frame again (this time we let the Artist watch)
    g_losses = criterion.generator_total(input_t, prev_t, generated, target_t, pred_fake_for_G) # The Grader calculates how badly the Artist failed
    g_losses["total"].backward() # Calculate the exact math adjustments needed to make the Artist a better drawer
    optG.step() # Apply the adjustments to the Artist's brain
```

---

## 4. How All Files Connect

```text
[ Raw Video ] --> (dataloader.py) --resizes and finds motion--> [ Math Tensor (Image + Flow) ]
                                                                             |
                                                                             v
                                                                      (generator.py)
                                                                             |
                                                                             v
                               [ Generated Fake Image ] <------------- [ Draws a new frame ]
                                         |
                                         v
 [ Real Video Frame ] ---------> (discriminator.py) <---------- [ Judges if Fake is actually Real ]
                                         |
                                         v
                                     (loss.py) <--------------- [ Grades how everyone performed ]
                                         |
                                         v
                                     (train.py) <-------------- [ Updates brains based on grades ]
                                         |
                                         v
                     (evaluate.py saves the day at the end!)
```

---

## 5. The Training Story
When we hit run, `dataloader.py` grabs a batch of 16 normal video frames (say, people walking on a sidewalk). It calculates how they are moving and hands this data to `generator.py`. The Generator attempts to draw what the current frame should look like. Because it just started, the drawing looks like blurry static. `discriminator.py` looks at the real video frame, then looks at the blurry static, and easily declares the static as "Fake". `loss.py` gives the Discriminator a good grade and the Generator a terrible grade. `train.py` takes these grades and slightly tweaks the math inside both brains. Millions of frames later, the Generator gets so incredibly good at drawing people walking on sidewalks that the Discriminator literally cannot tell the difference between the real video and the generated one!

---

## 6. The Detection Story
Now the training is over. The Discriminator goes on vacation; we don't need it anymore. We deploy our master Generator into a live security camera via `evaluate.py`. The camera sees a normal person walking. The Generator perfectly predicts and draws the person walking. The mathematical difference (MSE error) between the real camera and the drawing is 0.01 (Very Low). Suddenly, a skateboarder zooms across the screen! The Generator has *never* seen a skateboarder, so it draws a weird, blurry blob trying to make it look like a walking person. We compare the real skateboarder to the weird blob. The mathematical difference is 0.85 (Very High!). Because this difference crosses our `ANOMALY_THRESHOLD`, the system flashes red and flags the skateboarder as an anomaly!

---

## 7. The Three Losses Explained Simply

Think of the Generator taking a final exam with 3 different teachers grading it.

**1. Adversarial Loss (The Psychology Teacher):**
*The Analogy:* The teacher doesn't care about facts; they just care if you can confidently lie and sound like a human. They score you based on how well you tricked the lie-detector (Discriminator).
```python
def generator_loss(self, pred_fake: torch.Tensor) -> torch.Tensor: # Calculate how well we tricked the Critic
    target_real = torch.ones_like(pred_fake) # Create a fake scorecard filled with perfect "100%" scores
    return self.mse(pred_fake, target_real) # Grade the Artist based on how close the Critic's score was to 100%
```

**2. Reconstruction Loss (The Math Teacher):**
*The Analogy:* The teacher puts your drawing over the real photograph on a light-table. Every single pixel that is a different color loses you a point. It forces your drawing to be physically identical to reality.
```python
def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor: # Compare the drawing to the real frame
    return self.criterion(generated, target) # Calculate the exact absolute mathematical difference between every single pixel (L1 Loss)
```

**3. Warping Flow Loss (The Physics Teacher):**
*The Analogy:* The teacher says, "If you drew a car moving right at 50mph, and I digitally push the old video frame to the right at 50mph, they should match perfectly." It forces the AI to obey the laws of physics and actual movement.
```python
def forward(self, prev_frame: torch.Tensor, flow_uv: torch.Tensor, generated_frame: torch.Tensor) -> torch.Tensor: # Check if physics was obeyed
    B, C, H, W = prev_frame.shape # Find out how tall and wide the image is
    y_grid, x_grid = torch.meshgrid(...) # Create an invisible graph paper over the image
    grid = torch.stack([x_grid, y_grid], dim=-1) # Bundle the graph paper into a math object
    flow_offsets = flow_uv.permute(0, 2, 3, 1) # Prepare the movement data to be added to the graph paper
    warp_grid = grid + flow_offsets # Physically bend the graph paper based on where things are moving
    warped_prev = torch.nn.functional.grid_sample(prev_frame, warp_grid) # Take the old frame and smear its pixels across the bent graph paper
    return torch.nn.functional.l1_loss(warped_prev, generated_frame) # Check if our smeared old frame perfectly matches the new drawn frame
```

---

## 8. Glossary

*   **GAN (Generative Adversarial Network):** Two AIs fighting each other to get better. *Analogy:* An art forger (Generator) trying to paint fake Monets, and a police detective (Discriminator) trying to spot the fakes.
*   **U-Net:** A specific AI brain shape that shrinks an image to understand it, then blows it back up to draw it. *Analogy:* Taking apart a puzzle to see the pieces, then putting it back together.
*   **PatchGAN:** A critic that judges paintings by looking at them through a toilet paper tube. *Analogy:* Instead of saying "The whole painting is fake," it says "This specific 30x30 corner looks fake."
*   **LSGAN (Least Squares GAN):** A mathematical rule to prevent the AIs from getting stuck. *Analogy:* Using a ruler to measure mistakes instead of just passing/failing them.
*   **Optical Flow:** The tracking of how pixels move between two video frames. *Analogy:* Drawing arrows on a picture to show which way the wind is blowing the leaves.
*   **Farneback:** A specific, famous math equation used to calculate Optical Flow. *Analogy:* A specific brand of highly accurate radar gun used by police.
*   **AUC-ROC:** A score from 0 to 1 measuring how perfectly the AI detects anomalies without false alarms. *Analogy:* A batting average for baseball—1.0 means you hit every ball perfectly.
*   **L1 Loss:** Measuring a mistake by taking the absolute distance between a guess and the truth. *Analogy:* Guessing a person's age. If they are 20 and you guess 25, your L1 mistake is 5.
*   **Skip Connections:** Wires in the AI brain that skip the deep thinking and pass high-res details straight to the output. *Analogy:* Giving an artist a tracing-paper outline so they don't mess up the proportions.
*   **Instance Normalization:** Ensuring colors/brightness aren't too crazy before doing math on them. *Analogy:* Forcing everyone in a choir to sing at exactly the same volume so no one drowns out the others.
*   **grid_sample:** A PyTorch tool that moves pixels around based on a map. *Analogy:* Printing a picture on silly putty and stretching it.
*   **Anomaly Score:** A single number representing how "weird" a video frame is. *Analogy:* A Geiger counter clicking faster when radiation is nearby.
*   **Temporal Smoothing:** Averaging out scores over time to ignore random 1-frame glitches. *Analogy:* Ignoring a single weird noise in the house, but calling the cops if the noise continues for 15 seconds.
