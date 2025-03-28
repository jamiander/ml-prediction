import requests
import matplotlib.pyplot as plt
import numpy as np

# Fetch data from the server
response = requests.get('http://localhost:5000/plot')
data = response.json()

if data['status'] == 'success':
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Define colors and styles for different combinations
    styles = {
        'employed': {
            'married': {'color': 'darkred', 'linestyle': '-', 'alpha': 1.0},
            'unmarried': {'color': 'red', 'linestyle': '--', 'alpha': 0.8}
        },
        'unemployed': {
            'married': {'color': 'darkblue', 'linestyle': '-', 'alpha': 1.0},
            'unmarried': {'color': 'blue', 'linestyle': '--', 'alpha': 0.8}
        }
    }
    
    # First subplot: Default age (20)
    curves = data['curves']
    
    # Plot each curve for age 20
    for emp_status in ['employed', 'unemployed']:
        for mar_status in ['married', 'unmarried']:
            points = curves[emp_status][mar_status]
            if points:
                grades = [p['grade'] for p in points]
                probs = [p['probability'] for p in points]
                style = styles[emp_status][mar_status]
                r_squared = data['stats'][emp_status][mar_status]['rSquared']
                label = f'{emp_status.title()} & {mar_status.title()} (R² = {r_squared:.3f})'
                ax1.plot(grades, probs, linewidth=2, label=label, **style)
    
    # Plot actual data points with jitter
    actual_points = data['actualPoints']
    jitter_amount = 0.02
    
    # Define scatter plot styles
    scatter_styles = {
        'employed': {
            'married': {'color': 'darkred', 'marker': 'o', 'alpha': 0.5, 's': 100},
            'unmarried': {'color': 'red', 'marker': 's', 'alpha': 0.5, 's': 100}
        },
        'unemployed': {
            'married': {'color': 'darkblue', 'marker': 'o', 'alpha': 0.5, 's': 100},
            'unmarried': {'color': 'blue', 'marker': 's', 'alpha': 0.5, 's': 100}
        }
    }
    
    # Plot actual points for each combination
    for emp_status in ['employed', 'unemployed']:
        for mar_status in ['married', 'unmarried']:
            points = [(p['grade'], p['graduate']) for p in actual_points 
                     if p['employed'] == (1 if emp_status == 'employed' else 0) and 
                        p['married'] == (1 if mar_status == 'married' else 0)]
            if points:
                grades, grad = zip(*points)
                grad_jittered = [g + np.random.uniform(-jitter_amount, jitter_amount) for g in grad]
                style = scatter_styles[emp_status][mar_status]
                label = f'{emp_status.title()} & {mar_status.title()} (actual)'
                ax1.scatter(grades, grad_jittered, label=label, **style)
    
    ax1.set_xlabel('GPA', fontsize=12)
    ax1.set_ylabel('Graduation Probability', fontsize=12)
    ax1.set_title('Graduation Probability vs GPA by Employment and Marital Status (Age 20)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_ylim(-0.15, 1.15)
    ax1.set_xlim(0.9, 4.1)
    
    # Add reference lines to first subplot
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax1.axhline(y=1, color='k', linestyle='-', alpha=0.2)
    ax1.axvline(x=2.8, color='g', linestyle='--', alpha=0.5, label='Threshold (2.8)')
    ax1.axhspan(0.5, 1.0, color='g', alpha=0.1, label='Likely to Graduate')
    ax1.axhspan(0.0, 0.5, color='r', alpha=0.1, label='Unlikely to Graduate')
    
    # Second subplot: Age variation for a fixed GPA
    fixed_gpa = 3.0
    ages = np.linspace(18, 40, 50)
    
    # Get predictions for different ages
    age_predictions = {
        'employed': {
            'married': [],
            'unmarried': []
        },
        'unemployed': {
            'married': [],
            'unmarried': []
        }
    }
    
    # Make predictions for each age
    for age in ages:
        response = requests.get(f'http://localhost:5000/predict?grade={fixed_gpa}&age={age}')
        
        # Get predictions for all combinations
        for employed in [0, 1]:
            for married in [0, 1]:
                response = requests.get(
                    f'http://localhost:5000/predict?grade={fixed_gpa}&age={age}&employed={employed}&married={married}'
                )
                pred_data = response.json()
                emp_status = 'employed' if employed == 1 else 'unemployed'
                mar_status = 'married' if married == 1 else 'unmarried'
                age_predictions[emp_status][mar_status].append(pred_data['graduationProbability'])
    
    # Plot age variation curves
    for emp_status in ['employed', 'unemployed']:
        for mar_status in ['married', 'unmarried']:
            style = styles[emp_status][mar_status]
            label = f'{emp_status.title()} & {mar_status.title()}'
            ax2.plot(ages, age_predictions[emp_status][mar_status], 
                    linewidth=2, label=label, **style)
    
    # Plot actual points by age
    for emp_status in ['employed', 'unemployed']:
        for mar_status in ['married', 'unmarried']:
            points = [(p['age'], p['graduate']) for p in actual_points 
                     if p['employed'] == (1 if emp_status == 'employed' else 0) and 
                        p['married'] == (1 if mar_status == 'married' else 0) and
                        abs(p['grade'] - fixed_gpa) <= 0.2]  # Points near the fixed GPA
            if points:
                ages_data, grad = zip(*points)
                grad_jittered = [g + np.random.uniform(-jitter_amount, jitter_amount) for g in grad]
                style = scatter_styles[emp_status][mar_status]
                label = f'{emp_status.title()} & {mar_status.title()} (actual)'
                ax2.scatter(ages_data, grad_jittered, label=label, **style)
    
    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Graduation Probability', fontsize=12)
    ax2.set_title(f'Graduation Probability vs Age by Employment and Marital Status (GPA {fixed_gpa})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(-0.15, 1.15)
    ax2.set_xlim(18, 40)
    
    # Add reference lines to second subplot
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.axhline(y=1, color='k', linestyle='-', alpha=0.2)
    ax2.axhspan(0.5, 1.0, color='g', alpha=0.1, label='Likely to Graduate')
    ax2.axhspan(0.0, 0.5, color='r', alpha=0.1, label='Unlikely to Graduate')
    
    # Adjust layout
    plt.subplots_adjust(right=0.85, hspace=0.3)
    plt.show()
else:
    print("Error:", data.get('message', 'Unknown error')) 