const generatePlotData = (regressionModel, scaleFeatures) => {
    if (!regressionModel) {
        return {
            status: 'error',
            message: 'Model not trained'
        };
    }

    try {
        // Generate points for smooth curves
        const plotPoints = [];
        const defaultAge = 20;  // Use a default age for visualization
        
        for (let grade = 1.0; grade <= 4.0; grade += 0.1) {
            // Generate all polynomial features
            const features = scaleFeatures(grade, defaultAge);
            
            // Get predictions for both employed and unemployed
            let employedPred = null;
            let unemployedPred = null;

            if (regressionModel.models.employed) {
                const pred = regressionModel.models.employed.predict(features);
                employedPred = Number(Math.max(0, Math.min(1, Array.isArray(pred) ? pred[0] : pred)).toFixed(3));
            }

            if (regressionModel.models.unemployed) {
                const pred = regressionModel.models.unemployed.predict(features);
                unemployedPred = Number(Math.max(0, Math.min(1, Array.isArray(pred) ? pred[0] : pred)).toFixed(3));
            }

            plotPoints.push({
                grade: Number(grade.toFixed(1)),
                employed: employedPred,
                unemployed: unemployedPred
            });
        }

        return {
            status: 'success',
            plotPoints,
            modelStats: regressionModel.stats.modelStats
        };
    } catch (error) {
        return {
            status: 'error',
            message: 'Error generating plot data',
            error: error.message
        };
    }
};

export { generatePlotData }; 