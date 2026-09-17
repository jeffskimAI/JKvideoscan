### **Key Play Examples Across the Top 5 Sports**

| Sport | Routine Detections (Low Significance \~0.10–0.40) | Significant Key Plays (High Significance \~0.80–1.00) |
| :---- | :---- | :---- |
| **Tennis** | Standard rallies, faults, unforced errors, routine holds | Break point winners, tiebreak mini-breaks, match points, clutch aces |
| **Soccer** | Back passes, clearances, throw-ins, routine tackles | Stoppage-time goals, penalty saves, red cards, goal-line clearances |
| **Football** | 2-yard runs, incomplete screens, punts, routine tackles | Pick-sixes, 4th-quarter go-ahead TDs, red-zone strip sacks, walk-off FGs |
| **Baseball** | Routine balls/strikes, foul tips, routine pop-outs | Walk-off home runs, bases-loaded strikeouts, robbed home runs at the wall |
| **Boxing** | Single probing jabs, clinches, blocked punches | Knockdowns, referee 10-counts, fight-ending KOs/TKOs, staggering power flurries |

### 

### **How Gemini Distinguishes "Significant" Plays from 500 Detections**

A standard 2- to 3-hour live broadcast easily generates **500+ granular action detections** (routine passes, jabs, dribbles, foul balls, mid-court rallies, short rush gains).

To power member-facing features like **in-stream Key Plays carousels** or **automated recap reels**, Gemini separates two distinct scores:

> 1. **Detection Confidence** (e.g., 0.98): The probability that the event occurred and was classified correctly under Netflix’s canonical taxonomy.  
> 2. **Significance / Game-Impact Score** (e.g., 0.92 vs. 0.25): A composite editorial and match-leverage score evaluating:  
   * **Game State & Leverage**: Score delta, game clock, inning/quarter/set pressure (e.g., a 90th-minute equalizer ranks much higher than a goal scored while leading 4–0).  
   * **Audio & Atmosphere**: Multimodal spikes in crowd cheer, stadium decibels, and commentator excitement.  
   * **Visual & OCR Validation**: Scorebug changes, referee whistles, player celebrations, and on-screen graphics.  
   * **Event Rarity**: Low-frequency, game-altering actions (e.g., knockdowns, red cards, walk-offs, pick-sixes).

Each detected play is emitted with a **start/end timestamp**, a **recommended clip window** (buildup, peak event, celebration), **structured metadata** (players, score, period), and **explainability factors**.

### 

### **Details/Examples**

&nbsp;

#### **1\. 🎾 Tennis**

*Out of \~500 detections (serves, routine baseline groundstrokes, faults, defensive returns):*

* **Example A: Break Point Conversion via Passing Winner (tennis\_winner / tennis\_break\_point)**  
  * **Event**: In the 5th set at 4–4 (30–40), the receiving player executes a running down-the-line backhand winner to break serve.  
  * **Significance Factors**: High game leverage (break point in the final set), decisive score change, commentator volume peak, and crowd roar.  
  * **Clip Window**: \~15–20s (from the ball toss through the baseline rally, the winning shot, and fist-pump celebration).  
* **Example B: High-Pressure Ace on Match Point (tennis\_ace / tennis\_match\_point)**  
  * **Event**: Serving on championship point at 40–30, the server fires a 130 mph ace down the T to clinch the match.  
  * **Significance Factors**: Match-ending point context, decisive score impact, immediate referee "Game, Set, Match" announcement, and stadium ovation.

#### **2\. ⚽ Soccer**

*Out of \~500 detections (routine passes, throw-ins, clearances, off-target shots, mid-field tackles):*

* **Example A: Late Stoppage-Time Equalizer (soccer\_goal)**  
  * **Event**: At 90'+3, a corner kick is headed in to level the score at 2–2.  
  * **Significance Factors**: Game-saving score delta in stoppage time, explosive crowd acoustic spike, team dogpile celebration, and OCR scorebug update.  
  * **Clip Window**: \~25–35s (corner setup and kick delivery, header, goal confirmation, and celebration).  
* **Example B: Goalkeeper Penalty Kick Save (soccer\_penalty\_kick / soccer\_shot\_on\_target)**  
  * **Event**: In the 85th minute of a 1–0 match, the goalkeeper dives full extension to stop a penalty kick.  
  * **Significance Factors**: Win probability swing, high-drama set piece, instant visual/audio crowd reaction, and defensive stop under critical game pressure.

#### **3\. 🏈 American Football**

*Out of \~500 detections (huddles, routine 2-yard runs, incomplete passes, fair catches, standard punts):*

* **Example A: 4th-Quarter Interception Returned for a Touchdown (football\_interception / football\_touchdown)**  
  * **Event**: In Q4 with 1:45 remaining, trailing by 3, a cornerback intercepts an out-route and runs 45 yards for a "Pick-Six."  
  * **Significance Factors**: Immediate lead change in the final two minutes, extreme win-probability reversal, end-zone team celebration, and referee signaling.  
  * **Clip Window**: \~30–40s (pre-snap alignment, pass and interception, return to the end zone, and referee signal).  
* **Example B: 4th-and-Goal Strip Sack (football\_sack / football\_fumble\_recovery)**  
  * **Event**: On 4th-and-goal with 30 seconds left, an edge rusher blindsides the quarterback, forcing a loose ball recovered by the defense to seal the game.  
  * **Significance Factors**: Turnover on downs \+ turnover in the red zone, game-ending defensive stand, high broadcast replay frequency.

#### **4\. ⚾ Baseball**

*Out of \~500 detections (balls, called strikes, foul balls, routine infield groundouts, pop flies):*

* **Example A: Walk-Off Home Run (baseball\_home\_run / baseball\_run\_scored)**  
  * **Event**: Bottom of the 9th, 2 outs, tie game (3–3); batter hits a 420-foot walk-off blast into the outfield bleachers.  
  * **Significance Factors**: Game-ending outcome, maximum leverage index, home-plate mob celebration, pyrotechnics/horn audio triggers, and broadcast scorebug finalization.  
  * **Clip Window**: \~35–45s (pitch delivery, crack of the bat, home run trot, and home-plate celebration).  
* **Example B: Inning-Ending Bases-Loaded Strikeout (baseball\_strikeout)**  
  * **Event**: Top of the 8th, bases loaded, full count (3–2), pitcher throws a swinging strikeout to strand three runners and preserve a one-run lead.  
  * **Significance Factors**: High-leverage defensive escape (zero runs allowed with bases loaded), pitcher emotion/fist pump, and crowd ovation heading into the commercial break.

#### **5\. 🥊 Boxing (Combat Sports)**

*Out of \~500 detections (routine jabs, missed punches, parries, shoulder rolls, clinch resets):*

* **Example A: Decisive Knockdown (boxing\_knockdown / boxing\_power\_punch)**  
  * **Event**: In Round 7 of a title fight, a fighter lands a flush counter right cross that drops their opponent to the canvas.  
  * **Significance Factors**: Critical round scoring impact (10–8 round), referee actively initiating a 10-count, loud impact sound, and sudden commentator volume surge.  
  * **Clip Window**: \~25–30s (combination setup, clean punch connection, canvas fall, and referee count).  
* **Example B: Fight-Ending Knockout / TKO (boxing\_knockout / boxing\_technical\_knockout)**  
  * **Event**: In Round 10, an unanswered combination against the ropes causes the referee to step in and wave off the fight.  
  * **Significance Factors**: Match termination, change in championship title, visual intervention of the referee, corner celebrations, and official result confirmation.

&nbsp;