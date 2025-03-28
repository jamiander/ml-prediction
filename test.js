const generateTestData = (regressionModel, scaleFeatures) => {
    if (!regressionModel) {
        return {
            status: 'error',
            message: 'Model not trained or using interpolation',
            usingInterpolation: true
        };
    }

    try {
        // Test predictions on a range of grades with both employed states
        const testGrades = [1, 3, 3, 3.75, 4];
        const defaultAge = 20;
        const results = testGrades.flatMap(grade => {
            try {
                const results = [];
                const features = scaleFeatures(grade, defaultAge);
                
                if (regressionModel.models.unemployed) {
                    const unemployedPred = regressionModel.models.unemployed.predict(features);
                    const unemployedValue = Array.isArray(unemployedPred) ? unemployedPred[0] : unemployedPred;
                    results.push({
                        grade,
                        employed: 0,
                        age: defaultAge,
                        prediction: Number(Math.max(0, Math.min(1, unemployedValue)).toFixed(3))
                    });
                }
                
                if (regressionModel.models.employed) {
                    const employedPred = regressionModel.models.employed.predict(features);
                    const employedValue = Array.isArray(employedPred) ? employedPred[0] : employedPred;
                    results.push({
                        grade,
                        employed: 1,
                        age: defaultAge,
                        prediction: Number(Math.max(0, Math.min(1, employedValue)).toFixed(3))
                    });
                }
                
                return results;
            } catch (err) {
                console.error('Error predicting for grade:', grade, err);
                return [];
            }
        });

        return {
            status: 'success',
            modelTrained: true,
            testResults: results,
            modelStats: regressionModel.stats.modelStats
        };
    } catch (error) {
        return {
            status: 'error',
            message: 'Error testing model',
            error: error.message
        };
    }
};

export { generateTestData }; 