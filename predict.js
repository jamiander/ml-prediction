const getSuccessRate = (grade, employed = 0, age = 20, useInterpolation = false, regressionModel, scaleFeatures, studentData) => {
    const targetGrade = Number(grade);
    const targetAge = Number(age);
    const isEmployed = Number(employed);
    
    if (isNaN(targetGrade) || isNaN(targetAge)) {
        console.error('Invalid input:', { grade, age });
        return null;
    }
    
    // If model isn't trained yet or we're forced to use interpolation
    if (!regressionModel || useInterpolation) {
        console.log('Using interpolation method');
        // First try exact match
        const exactMatch = studentData.find(item => 
            Number(item.grade) === targetGrade && 
            Number(item.employed) === isEmployed &&
            Number(item.age) === targetAge
        );
        if (exactMatch) {
            console.log('Exact match found:', exactMatch);
            return Number(exactMatch.graduate);
        }

        // Sort data by grade for interpolation (matching employed status)
        const sortedData = studentData
            .filter(item => Number(item.employed) === isEmployed)
            .map(item => ({ 
                grade: Number(item.grade),
                age: Number(item.age), 
                graduate: Number(item.graduate)
            }))
            .sort((a, b) => a.grade - b.grade);

        if (sortedData.length === 0) {
            console.log('No data available for this employment status, using all data');
            // If no data for this employment status, use all data
            sortedData.push(...studentData
                .map(item => ({ 
                    grade: Number(item.grade),
                    age: Number(item.age), 
                    graduate: Number(item.graduate)
                }))
                .sort((a, b) => a.grade - b.grade)
            );
        }

        // Find closest grades
        let lowerGrade = null;
        let upperGrade = null;

        for (let i = 0; i < sortedData.length; i++) {
            if (sortedData[i].grade < targetGrade) {
                lowerGrade = sortedData[i];
            } else {
                upperGrade = sortedData[i];
                break;
            }
        }

        // Handle edge cases
        if (!lowerGrade && upperGrade) return Number(upperGrade.graduate.toFixed(3));
        if (!upperGrade && lowerGrade) return Number(lowerGrade.graduate.toFixed(3));
        if (!lowerGrade && !upperGrade) return 0.5; // Default value if no data available

        // Linear interpolation
        const gradeDiff = upperGrade.grade - lowerGrade.grade;
        const rateDiff = upperGrade.graduate - lowerGrade.graduate;
        const ratio = (targetGrade - lowerGrade.grade) / gradeDiff;
        
        const interpolatedValue = Number((lowerGrade.graduate + (rateDiff * ratio)).toFixed(3));
        console.log('Interpolation result:', {
            lowerGrade,
            upperGrade,
            gradeDiff,
            rateDiff,
            ratio,
            interpolatedValue
        });
        
        return interpolatedValue;
    }

    try {
        // Use ML model for prediction
        const model = isEmployed ? regressionModel.models.employed : regressionModel.models.unemployed;
        
        if (!model) {
            console.log('No model available for employment status:', isEmployed, 'using interpolation');
            return getSuccessRate(grade, employed, age, true, regressionModel, scaleFeatures, studentData);
        }

        // Generate all polynomial features
        const features = scaleFeatures(targetGrade, targetAge);

        console.log('Making prediction with features:', features, 'for employed:', isEmployed);
        
        const prediction = model.predict(features);
        const predictedValue = Array.isArray(prediction) ? prediction[0] : prediction;
        console.log('Raw prediction:', predictedValue);
        
        if (isNaN(predictedValue)) {
            console.log('Invalid prediction from model, using interpolation');
            return getSuccessRate(grade, employed, age, true, regressionModel, scaleFeatures, studentData);
        }
        
        // Normalize prediction between 0 and 1
        const normalizedPrediction = Number((Math.max(0, Math.min(1, predictedValue))).toFixed(3));
        console.log('Final prediction:', normalizedPrediction);
        
        return normalizedPrediction;
    } catch (error) {
        console.error('Error making prediction:', error.message);
        return getSuccessRate(grade, employed, age, true, regressionModel, scaleFeatures, studentData);
    }
};

export { getSuccessRate }; 