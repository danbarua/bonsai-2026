"""
Visualization utilities for learning tests.

This module provides utilities for visualizing the results of learning tests,
including character processing, phase patterns, and weight matrices.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .character_utils import get_character_matrix, calculate_local_coherence

# Try to import sklearn, but make it optional (matches the guard previously
# only present in the standalone tests/test_predictive_hebbian_character.py,
# ported here along with its bug fixes: SKLEARN_AVAILABLE needed a self.
# prefix there since it was a class attribute; here it's a plain module-level
# flag instead, and the TSNE perplexity is explicitly scaled to the sample
# count since the default of 30 is unreachable for small character sets.)
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Character embedding visualization will use a simple projection.")

# Ensure plots directory exists
os.makedirs('plots/hebbian', exist_ok=True)
os.makedirs('plots/predictive', exist_ok=True)

def visualize_character_state(state, weights, char, model_type='hebbian', save_path=None):
    """
    Visualize the state after processing a character.
    
    Args:
        state: The oscillator state
        weights: The weight matrix or dictionary of weight matrices
        char: The character that was processed
        model_type: The type of model ('hebbian' or 'predictive')
        save_path: Path to save the plot to. If None, uses default path.
    
    Returns:
        The coherence map for the character layer
    """
    # Extract phases from the first layer
    phases = state.phases[0]
    
    # Calculate phase coherence map
    coherence_map = calculate_local_coherence(phases)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original character
    axes[0, 0].imshow(get_character_matrix(char), cmap='binary')
    axes[0, 0].set_title(f"Original Character: '{char}'")
    
    # Phase map
    phase_img = axes[0, 1].imshow(phases, cmap='hsv')
    axes[0, 1].set_title("Final Phase Distribution")
    plt.colorbar(phase_img, ax=axes[0, 1])
    
    # Coherence map
    coh_img = axes[1, 0].imshow(coherence_map, cmap='viridis')
    axes[1, 0].set_title("Local Phase Coherence")
    plt.colorbar(coh_img, ax=axes[1, 0])
    
    # Weight matrix visualization
    if model_type == 'hebbian':
        # For Hebbian model, weights is a list of weight matrices
        if isinstance(weights, list):
            weight_matrix = weights[0]
        elif isinstance(weights, dict) and "weights" in weights:
            weight_matrix = weights["weights"][0]
        else:
            # Fallback: just use the first layer phases as a placeholder
            weight_matrix = np.outer(np.cos(phases.flatten()), np.cos(phases.flatten()))
    else:
        # For predictive model, weights is a dictionary
        if isinstance(weights, dict) and "within_layer_weights" in weights:
            weight_matrix = weights["within_layer_weights"][0]
        else:
            # Fallback: just use the first layer phases as a placeholder
            weight_matrix = np.outer(np.cos(phases.flatten()), np.cos(phases.flatten()))
    
    # Reshape weights to visualize connections between oscillators
    n_oscillators = np.prod(phases.shape)
    
    # Check if weight matrix needs reshaping and has the right size
    if weight_matrix.shape != (n_oscillators, n_oscillators):
        # If not the right shape, create a visualization based on phases
        weight_viz = np.outer(np.cos(phases.flatten()), np.cos(phases.flatten()))
        weight_img = axes[1, 1].imshow(weight_viz, cmap='coolwarm')
        axes[1, 1].set_title("Phase Correlation Matrix")
    else:
        # Use the actual weight matrix
        weight_img = axes[1, 1].imshow(weight_matrix, cmap='coolwarm')
        axes[1, 1].set_title("Final Weight Matrix")
    axes[1, 1].set_title("Final Weight Matrix")
    plt.colorbar(weight_img, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        # Default path based on model type
        save_path = f"plots/{model_type}/character_{char}_analysis.png"
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return coherence_map

def visualize_noisy_character(clean_state, noisy_state, char, noise_level, model_type='hebbian', save_path=None):
    """
    Visualize the processing of a noisy character.
    
    Args:
        clean_state: The state after processing the clean character
        noisy_state: The state after processing the noisy character
        char: The character that was processed
        noise_level: The noise level that was applied
        model_type: The type of model ('hebbian' or 'predictive')
        save_path: Path to save the plot to. If None, uses default path.
    """
    # Extract phases
    clean_phases = clean_state.phases[0]
    noisy_phases = noisy_state.phases[0]
    
    # Calculate coherence maps
    clean_coherence = calculate_local_coherence(clean_phases)
    noisy_coherence = calculate_local_coherence(noisy_phases)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original character
    axes[0, 0].imshow(get_character_matrix(char), cmap='binary')
    axes[0, 0].set_title(f"Original Character: '{char}'")
    
    # Noisy character phases
    phase_img = axes[0, 1].imshow(noisy_phases, cmap='hsv')
    axes[0, 1].set_title(f"Noisy Character Phases ({noise_level*100:.0f}% noise)")
    plt.colorbar(phase_img, ax=axes[0, 1])
    
    # Clean coherence map
    coh_img1 = axes[1, 0].imshow(clean_coherence, cmap='viridis')
    axes[1, 0].set_title("Clean Character Coherence")
    plt.colorbar(coh_img1, ax=axes[1, 0])
    
    # Noisy coherence map
    coh_img2 = axes[1, 1].imshow(noisy_coherence, cmap='viridis')
    axes[1, 1].set_title("Noisy Character Coherence")
    plt.colorbar(coh_img2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        # Default path based on model type
        save_path = f"plots/{model_type}/character_{char}_noisy_{int(noise_level*100)}.png"
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_model_comparison(char, hebbian_state, predictive_state, save_path=None):
    """
    Visualize a comparison between Hebbian and Predictive models.
    
    Args:
        char: The character that was processed
        hebbian_state: The state after processing with the Hebbian model
        predictive_state: The state after processing with the Predictive model
        save_path: Path to save the plot to. If None, uses default path.
    """
    # Extract phases
    hebbian_phases = hebbian_state.phases[0]
    predictive_phases = predictive_state.phases[0]
    
    # Calculate coherence maps
    hebbian_coherence = calculate_local_coherence(hebbian_phases)
    predictive_coherence = calculate_local_coherence(predictive_phases)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original character
    axes[0, 0].imshow(get_character_matrix(char), cmap='binary')
    axes[0, 0].set_title(f"Original Character: '{char}'")
    
    # Hebbian phases
    phase_img1 = axes[0, 1].imshow(hebbian_phases, cmap='hsv')
    axes[0, 1].set_title("Hebbian Model Phases")
    plt.colorbar(phase_img1, ax=axes[0, 1])
    
    # Predictive phases
    phase_img2 = axes[0, 2].imshow(predictive_phases, cmap='hsv')
    axes[0, 2].set_title("Predictive Model Phases")
    plt.colorbar(phase_img2, ax=axes[0, 2])
    
    # Phase difference
    phase_diff = np.abs(np.angle(np.exp(1j * (hebbian_phases - predictive_phases))))
    diff_img = axes[1, 0].imshow(phase_diff, cmap='viridis')
    axes[1, 0].set_title("Phase Difference")
    plt.colorbar(diff_img, ax=axes[1, 0])
    
    # Hebbian coherence
    coh_img1 = axes[1, 1].imshow(hebbian_coherence, cmap='viridis')
    axes[1, 1].set_title("Hebbian Coherence")
    plt.colorbar(coh_img1, ax=axes[1, 1])
    
    # Predictive coherence
    coh_img2 = axes[1, 2].imshow(predictive_coherence, cmap='viridis')
    axes[1, 2].set_title("Predictive Coherence")
    plt.colorbar(coh_img2, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f"plots/comparison/character_{char}_model_comparison.png"
        # Ensure directory exists
        os.makedirs('plots/comparison', exist_ok=True)
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_hierarchical_representation(state, char, save_path=None):
    """
    Visualize how a character is represented across the hierarchy of layers.
    
    Args:
        state: The hierarchical oscillator state
        char: The character that was processed
        save_path: Path to save the plot to. If None, uses default path.
    """
    # Number of layers
    n_layers = len(state.phases)
    
    # Create a figure with rows for different visualization types and columns for layers
    fig, axes = plt.subplots(4, n_layers, figsize=(n_layers*4, 16))
    
    # If only one layer, reshape axes for consistent indexing
    if n_layers == 1:
        axes = axes.reshape(4, 1)
    
    # Original character for reference (first column, first row)
    char_matrix = get_character_matrix(char)
    axes[0, 0].imshow(char_matrix, cmap='binary')
    axes[0, 0].set_title(f"Original Character: '{char}'")
    
    # For each layer, show different visualizations
    for i in range(n_layers):
        layer_name = state.layer_names[i]
        phase_data = state.phases[i]
        
        # Row 1: Phase distribution using hsv colormap (circular)
        if i > 0:  # Skip first column of first row (used for original character)
            axes[0, i].imshow(phase_data, cmap='hsv')
            axes[0, i].set_title(f"Phase Distribution\n{layer_name}")
        
        # Row 2: Local coherence map
        coherence_map = calculate_local_coherence(phase_data)
        coh_img = axes[1, i].imshow(coherence_map, cmap='viridis')
        axes[1, i].set_title(f"Local Coherence\n{layer_name}")
        plt.colorbar(coh_img, ax=axes[1, i])
        
        # Row 3: Phase gradient magnitude (spatial derivative)
        # This shows where phase changes rapidly vs. smoothly
        gradient_y, gradient_x = np.gradient(phase_data)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        grad_img = axes[2, i].imshow(gradient_mag, cmap='magma')
        axes[2, i].set_title(f"Phase Gradient\n{layer_name}")
        plt.colorbar(grad_img, ax=axes[2, i])
        
        # Row 4: Oscillator activity (complex representation)
        # Convert phases to complex numbers and visualize magnitude/angle
        complex_z = np.exp(1j * phase_data)
        activity = np.abs(complex_z)  # Should be 1 everywhere, but useful for verification
        act_img = axes[3, i].imshow(activity, cmap='plasma')
        axes[3, i].set_title(f"Oscillator Activity\n{layer_name}")
        plt.colorbar(act_img, ax=axes[3, i])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f"plots/predictive/hierarchical_{char}_representation.png"
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_feature_extraction(state, weights, char, save_path=None):
    """
    Visualize how features are extracted and transformed between layers.
    
    Args:
        state: The hierarchical oscillator state
        weights: Dictionary of weight matrices
        char: The character that was processed
        save_path: Path to save the plot to. If None, uses default path.
    """
    n_layers = len(state.phases)
    
    # Create a figure with 2 rows: top for phase patterns, bottom for weight visualizations
    fig, axes = plt.subplots(2, n_layers-1, figsize=((n_layers-1)*5, 10))
    
    # If only one layer pair, reshape axes for consistent indexing
    if n_layers == 2:
        axes = axes.reshape(2, 1)
    
    # For each pair of adjacent layers
    for i in range(n_layers-1):
        # Get the between-layer weights
        between_weights = weights["between_layer_weights"][i]
        
        # Reshape weights for visualization if needed
        # This depends on how the weights are stored and the layer shapes
        lower_shape = state.layer_shapes[i]
        higher_shape = state.layer_shapes[i+1]
        
        # Top row: Show the phase patterns of adjacent layers
        lower_phases = state.phases[i]
        higher_phases = state.phases[i+1]
        
        # Create a side-by-side comparison
        comparison = np.zeros((max(lower_shape[0], higher_shape[0]), 
                              lower_shape[1] + higher_shape[1]))
        
        # Insert the phase patterns
        comparison[:lower_shape[0], :lower_shape[1]] = lower_phases
        comparison[:higher_shape[0], lower_shape[1]:] = higher_phases
        
        # Display the comparison
        phase_img = axes[0, i].imshow(comparison, cmap='hsv')
        axes[0, i].set_title(f"Phase Patterns: {state.layer_names[i]} → {state.layer_names[i+1]}")
        plt.colorbar(phase_img, ax=axes[0, i])
        
        # Add a vertical line to separate the layers
        axes[0, i].axvline(x=lower_shape[1]-0.5, color='white', linestyle='-', linewidth=2)
        
        # Bottom row: Visualize the weight matrix using SVD to find principal components
        try:
            u, s, vh = np.linalg.svd(between_weights, full_matrices=False)
            
            # Use top 2 components to create a 2D visualization of weight space
            weight_viz = np.outer(u[:, 0], vh[0, :]) * s[0]
            if s.size > 1:  # Add second component if available
                weight_viz += np.outer(u[:, 1], vh[1, :]) * s[1]
            
            # Reshape to match layer dimensions if possible
            try:
                weight_viz_reshaped = weight_viz.reshape(higher_shape[0], higher_shape[1], 
                                                       lower_shape[0], lower_shape[1])
                # Average across input dimensions to get a 2D map
                weight_map = np.mean(weight_viz_reshaped, axis=(2, 3))
                weight_img = axes[1, i].imshow(weight_map, cmap='coolwarm')
                axes[1, i].set_title(f"Weight Principal Components\n{state.layer_names[i]} → {state.layer_names[i+1]}")
            except:
                # Fallback: just show the raw weight matrix
                weight_img = axes[1, i].imshow(weight_viz, cmap='coolwarm')
                axes[1, i].set_title(f"Weight Matrix\n{state.layer_names[i]} → {state.layer_names[i+1]}")
        except:
            # Fallback: just show the raw weight matrix
            weight_img = axes[1, i].imshow(between_weights, cmap='coolwarm')
            axes[1, i].set_title(f"Weight Matrix\n{state.layer_names[i]} → {state.layer_names[i+1]}")
        
        plt.colorbar(weight_img, ax=axes[1, i])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f"plots/predictive/hierarchical_{char}_features.png"
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_reconstruction(state, weights, char, save_path=None):
    """
    Visualize how well each layer can reconstruct the original character.
    
    Args:
        state: The hierarchical oscillator state
        weights: Dictionary of weight matrices
        char: The character that was processed
        save_path: Path to save the plot to. If None, uses default path.
    """
    n_layers = len(state.phases)
    
    # Create a figure
    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers*4, 4))
    
    # If only one layer, reshape axes for consistent indexing
    if n_layers == 1:
        axes = [axes]
    
    # Original character
    char_matrix = get_character_matrix(char)
    axes[0].imshow(char_matrix, cmap='binary')
    axes[0].set_title(f"Original: '{char}'")
    
    # For each higher layer, attempt reconstruction
    for i in range(1, n_layers):
        # Get the phase pattern for this layer
        higher_phases = state.phases[i]
        
        # Convert to complex representation
        higher_complex = np.exp(1j * higher_phases.flatten())
        
        # Reconstruct through each layer back to the input
        reconstructed = higher_complex
        for j in range(i-1, -1, -1):
            # Use transpose of weights for backward projection
            between_weights = weights["between_layer_weights"][j]
            reconstructed = between_weights.T @ reconstructed
            
            # Normalize to unit circle
            reconstructed = reconstructed / np.abs(reconstructed)
        
        # Convert back to phase representation
        reconstructed_phases = np.angle(reconstructed)
        
        # Reshape to match input dimensions
        try:
            reconstructed_phases = reconstructed_phases.reshape(state.layer_shapes[0])
        except:
            # If reshaping fails, use a default shape
            reconstructed_phases = reconstructed_phases.reshape(char_matrix.shape)
        
        # Display the reconstruction
        axes[i].imshow(reconstructed_phases, cmap='hsv')
        axes[i].set_title(f"Reconstructed from Layer {i+1}")
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f"plots/predictive/hierarchical_{char}_reconstruction.png"
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_ambiguity_resolution(ambiguous_matrix, predictive_state, hebbian_state, char1, char2, ambiguity_level, save_path=None):
    """
    Visualize how ambiguous characters are resolved by different models.
    
    Args:
        ambiguous_matrix: The ambiguous character matrix
        predictive_state: State after processing with predictive Hebbian model
        hebbian_state: State after processing with standard Hebbian model
        char1, char2: The two characters being blended
        ambiguity_level: The level of ambiguity applied
        save_path: Path to save the plot to. If None, uses default path.
    """
    # Create a figure with 2 rows: top for characters, bottom for phase patterns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original characters
    char1_matrix = get_character_matrix(char1)
    axes[0, 0].imshow(char1_matrix, cmap='binary')
    axes[0, 0].set_title(f"Character 1: '{char1}'")
    
    char2_matrix = get_character_matrix(char2)
    axes[0, 1].imshow(char2_matrix, cmap='binary')
    axes[0, 1].set_title(f"Character 2: '{char2}'")
    
    # Ambiguous character
    axes[0, 2].imshow(ambiguous_matrix, cmap='binary')
    axes[0, 2].set_title(f"Ambiguous Character\n({ambiguity_level*100:.0f}% blend)")
    
    # Difference map between original characters
    diff_map = np.abs(char1_matrix - char2_matrix)
    axes[0, 3].imshow(diff_map, cmap='binary')
    axes[0, 3].set_title("Difference Map")
    
    # Display phase patterns with similarity scores
    axes[1, 0].imshow(predictive_state.phases[0], cmap='hsv')
    axes[1, 0].set_title(f"Predictive Model Result")
    
    axes[1, 1].imshow(hebbian_state.phases[0], cmap='hsv')
    axes[1, 1].set_title(f"Hebbian Model Result")
    
    # Calculate coherence maps
    pred_coherence = calculate_local_coherence(predictive_state.phases[0])
    hebb_coherence = calculate_local_coherence(hebbian_state.phases[0])
    
    # Display coherence maps
    coh_img1 = axes[1, 2].imshow(pred_coherence, cmap='viridis')
    axes[1, 2].set_title("Predictive Coherence")
    plt.colorbar(coh_img1, ax=axes[1, 2])
    
    coh_img2 = axes[1, 3].imshow(hebb_coherence, cmap='viridis')
    axes[1, 3].set_title("Hebbian Coherence")
    plt.colorbar(coh_img2, ax=axes[1, 3])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        # Ensure directory exists
        os.makedirs('plots/comparison', exist_ok=True)
        save_path = f"plots/comparison/ambiguity_{char1}_{char2}_{int(ambiguity_level*100)}.png"
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def visualize_occlusion_handling(clean_state, occluded_state, predictive_state, hebbian_state, char, occlusion_type, occlusion_level, save_path=None):
    """
    Visualize handling of occluded characters by different models.
    
    Args:
        clean_state: Original clean character state
        occluded_state: Occluded character state
        predictive_state: State after processing with predictive model
        hebbian_state: State after processing with hebbian model
        char: The character being processed
        occlusion_type: Type of occlusion applied
        occlusion_level: Level of occlusion applied
        save_path: Path to save the plot to. If None, uses default path.
    """
    # Create a figure with 2 rows: top for input/output, bottom for coherence maps
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original character
    char_matrix = get_character_matrix(char)
    axes[0, 0].imshow(char_matrix, cmap='binary')
    axes[0, 0].set_title(f"Original Character: '{char}'")
    
    # Occluded character
    occluded_matrix = occluded_state.perturbations[0] / np.max(occluded_state.perturbations[0])
    axes[0, 1].imshow(occluded_matrix, cmap='binary')
    axes[0, 1].set_title(f"Occluded ({occlusion_type}, {occlusion_level*100:.0f}%)")
    
    # Predictive model result
    axes[0, 2].imshow(predictive_state.phases[0], cmap='hsv')
    axes[0, 2].set_title("Predictive Result")
    
    # Hebbian model result
    axes[0, 3].imshow(hebbian_state.phases[0], cmap='hsv')
    axes[0, 3].set_title("Hebbian Result")
    
    # Coherence maps
    # Clean state coherence
    clean_coherence = calculate_local_coherence(clean_state.phases[0])
    coh_img1 = axes[1, 0].imshow(clean_coherence, cmap='viridis')
    axes[1, 0].set_title("Clean Coherence")
    plt.colorbar(coh_img1, ax=axes[1, 0])
    
    # Occluded state coherence
    occluded_coherence = calculate_local_coherence(occluded_state.phases[0])
    coh_img2 = axes[1, 1].imshow(occluded_coherence, cmap='viridis')
    axes[1, 1].set_title("Occluded Coherence")
    plt.colorbar(coh_img2, ax=axes[1, 1])
    
    # Predictive model coherence
    predictive_coherence = calculate_local_coherence(predictive_state.phases[0])
    coh_img3 = axes[1, 2].imshow(predictive_coherence, cmap='viridis')
    axes[1, 2].set_title("Predictive Coherence")
    plt.colorbar(coh_img3, ax=axes[1, 2])
    
    # Hebbian model coherence
    hebbian_coherence = calculate_local_coherence(hebbian_state.phases[0])
    coh_img4 = axes[1, 3].imshow(hebbian_coherence, cmap='viridis')
    axes[1, 3].set_title("Hebbian Coherence")
    plt.colorbar(coh_img4, ax=axes[1, 3])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        # Ensure directory exists
        os.makedirs('plots/comparison', exist_ok=True)
        save_path = f"plots/comparison/occlusion_{char}_{occlusion_type}_{int(occlusion_level*100)}.png"
    
    # Ensure the output directory exists regardless of whether save_path was
    # the function's own default or supplied by the caller -- previously this
    # only happened in some functions' if-save_path-is-None branch, so any
    # caller-supplied path into a not-yet-existing directory (e.g. plots/comparison/)
    # failed with FileNotFoundError.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def visualize_character_embedding(character_states, characters, save_path=None):
    """
    Visualize how different characters are embedded in the phase space of each
    layer, using dimensionality reduction (PCA or t-SNE) to create 2D plots.

    Ported from tests/test_predictive_hebbian_character.py (the standalone
    duplicate test file, retired in favor of this shared viz_utils module),
    with two bugs already fixed there carried over:
    - t-SNE's perplexity is explicitly capped at len(characters)-1, since the
      default of 30 is unreachable for small character sets (previously
      caused a NameError from an unrelated missing 'self.' typo that masked
      this issue, then a separate sklearn ValueError once that was fixed).
    - No 'self.SKLEARN_AVAILABLE' class-attribute indirection -- this module
      uses a plain module-level SKLEARN_AVAILABLE flag instead.

    Args:
        character_states: dict mapping character -> its final LayeredOscillatorState
        characters: list of characters that were processed
        save_path: output path for the figure
    """
    first_char = characters[0]
    n_layers = len(character_states[first_char].phases)

    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 5, 5))
    if n_layers == 1:
        axes = [axes]

    for layer_idx in range(n_layers):
        phase_data = []
        for char in characters:
            flat_phases = character_states[char].phases[layer_idx].flatten()
            complex_phases = np.exp(1j * flat_phases)
            features = np.concatenate([complex_phases.real, complex_phases.imag])
            phase_data.append(features)
        phase_data = np.array(phase_data)

        if SKLEARN_AVAILABLE:
            if len(characters) >= 5:
                tsne = TSNE(n_components=2, random_state=42,
                            perplexity=min(30, len(characters) - 1))
                embedding = tsne.fit_transform(phase_data)
                method = "t-SNE"
            else:
                pca = PCA(n_components=2)
                embedding = pca.fit_transform(phase_data)
                method = "PCA"

            for i, char in enumerate(characters):
                axes[layer_idx].scatter(embedding[i, 0], embedding[i, 1], s=100, label=char)
            axes[layer_idx].set_title(f"Layer {layer_idx+1} Character Embedding ({method})")
        else:
            for i, char in enumerate(characters):
                complex_phases = np.exp(1j * character_states[char].phases[layer_idx].flatten())
                if len(complex_phases) >= 2:
                    x, y = complex_phases[0].real, complex_phases[1].real
                    axes[layer_idx].scatter(x, y, s=100, label=char)
                else:
                    axes[layer_idx].scatter(i, 0, s=100, label=char)
            axes[layer_idx].set_title(f"Layer {layer_idx+1} Character Embedding (Simple Projection)")

        axes[layer_idx].legend()
        axes[layer_idx].grid(True)

    plt.tight_layout()
    if save_path is None:
        save_path = "plots/comparison/character_embedding.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
