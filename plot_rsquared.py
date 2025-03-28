import requests
import matplotlib.pyplot as plt
import numpy as np

# Fetch data from the server
response = requests.get('http://localhost:5000/plot')
data = response.json()

if data['status'] == 'success':
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the fitted curves
    plot_points = data['plotPoints']
    grades = [p['grade'] for p in plot_points]
    
    # Plot employed curve if available
    employed_pred = [p['employed'] for p in plot_points]
    if any(pred is not None for pred in employed_pred):
        plt.plot(grades, employed_pred, 'r-', linewidth=2, 
                label=f'Employed (R² = {data["modelStats"]["employed"]["rSquared"]:.3f})')
    
    # Plot unemployed curve if available
    unemployed_pred = [p['unemployed'] for p in plot_points]
    if any(pred is not None for pred in unemployed_pred):
        plt.plot(grades, unemployed_pred, 'b-', linewidth=2,
                label=f'Unemployed (R² = {data["modelStats"]["unemployed"]["rSquared"]:.3f})')
    
    # Plot actual data points with jitter
    actual_points = data['actualPoints']
    jitter_amount = 0.02  # Amount of vertical jitter
    
    # Employed points
    employed_points = [(p['grade'], p['graduate']) for p in actual_points if p['employed'] == 1]
    if employed_points:
        grades_emp, grad_emp = zip(*employed_points)
        # Add jitter to graduation values
        grad_emp_jittered = [g + np.random.uniform(-jitter_amount, jitter_amount) for g in grad_emp]
        plt.scatter(grades_emp, grad_emp_jittered, c='red', marker='o', alpha=0.5, 
                   label='Employed (actual)', s=100)
    
    # Unemployed points
    unemployed_points = [(p['grade'], p['graduate']) for p in actual_points if p['employed'] == 0]
    if unemployed_points:
        grades_unemp, grad_unemp = zip(*unemployed_points)
        # Add jitter to graduation values
        grad_unemp_jittered = [g + np.random.uniform(-jitter_amount, jitter_amount) for g in grad_unemp]
        plt.scatter(grades_unemp, grad_unemp_jittered, c='blue', marker='o', alpha=0.5, 
                   label='Unemployed (actual)', s=100)
    
    plt.xlabel('GPA', fontsize=12)
    plt.ylabel('Graduation Probability', fontsize=12)
    plt.title('Graduation Probability vs GPA with R-squared Fit', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Set y-axis limits with some padding for jitter
    plt.ylim(-0.15, 1.15)
    plt.xlim(0.9, 4.1)
    
    # Add horizontal lines at 0 and 1
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axhline(y=1, color='k', linestyle='-', alpha=0.2)
    
    # Add vertical line at GPA threshold with label
    plt.axvline(x=2.8, color='g', linestyle='--', alpha=0.5, 
                label='Threshold (2.8)')
    
    # Add shaded regions for different probability zones
    plt.axhspan(0.5, 1.0, color='g', alpha=0.1, label='Likely to Graduate')
    plt.axhspan(0.0, 0.5, color='r', alpha=0.1, label='Unlikely to Graduate')
    
    plt.tight_layout()
    plt.show()
else:
    print("Error:", data.get('message', 'Unknown error')) 