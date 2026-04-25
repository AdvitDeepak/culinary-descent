# Culinary Design for Descent: Differentiable Program Discovery for Procedural Recipes (CS 348K Project)

Name: Advit Deepak (advit)

# Summary

We build a system that can learn the specific ingredients and cooking steps (“culinary program”) needed to achieve a target visual outcome. We define a procedural grammar for cooking based on the work in [Learning Program Representations for Food Images and Cooking Recipes (MIT, CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Papadopoulos_Learning_Program_Representations_for_Food_Images_and_Cooking_Recipes_CVPR_2022_paper.pdf). Then, we modify the framework of the paper we covered in class, [Design for Descent (SIGA 2025)](https://www.computationaldesign.group/assets/papers/SIGA-2025-D4Descent.pdf), to work for this grammar given its inverse nature (_as now, multiple ingredients (roots) converge into a single dish (leaf) in the grammar_). By using visual similarity signal as the objective (leveraging the MIT work’s recipe2image models), we can use a variant of Stochastic Rewrite Descent to find the optimal recipe given a target image.

# Inputs / Outputs 

Inputs:
* A target image $I_{target}$ of a desired dish (food item)
* A procedural grammar $\mathcal{G}$ defining valid ingredient combinations and atomic actions (e.g., sauté, blend, sear).
* A pre-trained recipe2image model (taken from the MIT paper) to act as the differentiable renderer.

Outputs: 
* A generated Recipe DAG (the "Culinary Program") representing the discovered and optimal steps.
* The resulting image $I_{gen}$ synthesized from applying the recipe2image model on the program.
* Optimization curves showing improvement over iterations.

Design Constraints: 
* Design for Descent's grammars all expand from one seed/root to many parts ($1 \to N$). Recipes are inherently convergent -- multiple ingredients (roots) merge into a single dish (leaf). As a result, we must modify the rewrite logic to handle this "many-to-one" material flow.


# Task List


* Baseline & Environment: Set up the MIT CVPR 2022 codebase on a DGX node. Verify the recipe2image model by generating images from hand-written programs (e.g., "omelette").

* Grammar Design: Define a DSL for recipes using the CVPR 2022 vocabulary. Implement the DAG-to-Program translation layer.

* Differentiable Loop: Integrate the D4: Descent logic. This involves implementing the stochastic rewrite descent to handle the inverse DAG structure, allowing gradients to flow from the final image back to the ingredient selection.

* Optimization: Can try implementing a parallel sampling kernel to evaluate multiple recipe candidates in a single batch.

## Milestones

* Week 5-6: Recreated both repos, generated an image from a hard-coded DAG, and establish a random-search baseline.

* Week 7-8: Completed the modified D4 descent algorithm, and "solved" for simple recipes given output images.

* Week 9-10: Scale to complex dishes, optimize algorithm (parallel sampling), compare to other methods. 

Nice-to-have: Implement cooking constraints as part of the grammar to ensure the discovered programs aren't just visually correct but follow logical cooking sequences (e.g., no "blending" after "plating").

# Deliverables

The primary deliverable is a "Culinary Search Engine" demo. We will show a sequence of "target images" alongside the "discovered recipes" the system generated as well as the official recipes which correspond to the images.

Our evaluation metrics will include:

* Visual Fidelity: Similarity scores between the target and the generated output.

* Optimization Efficiency: A graph showing the number of iterations/samples required for convergence

* Ablation Study: A comparison of how parts of the grammar and the D4Descent authors' four principles affect results.

# Biggest Risks

One of the biggest risks would be correcting the D4 logic to handle $N \to 1$ structures. If the "inverse" logic becomes unstable, we may have to reformulate the problem as a sequence-generation task with fixed slots. Additionally, the set of state transformations (procedures/ingredients) might be too large, in which case filtering of the data will be required.

# Help / Advice

Would love any pointers on how to handle these inverse/convergent graph structures (many roots leading to one leaf)!

