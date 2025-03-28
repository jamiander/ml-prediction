const generateTestData = (regressionModel, scaleFeatures) => {
    if (!regressionModel) {
        return {
            status: 'error',
            message: 'Model not trained',
            usingInterpolation: true
        };
    }

    try {
        // Test grades
        const grades = [1, 3, 3.3, 3.75, 4];
        const defaultAge = 20;
        const results = [];

        // Test each grade for all combinations of employed and married status
        for (const grade of grades) {
            for (const employed of [0, 1]) {
                for (const married of [0, 1]) {
                    const features = scaleFeatures(grade, defaultAge, married);
                    const model = employed ? regressionModel.models.employed : regressionModel.models.unemployed;
                    
                    if (!model) {
                        console.log('No model available for employment status:', employed);
                        continue;
                    }

                    const prediction = model.predict(features);
                    const predictedValue = Array.isArray(prediction) ? prediction[0] : prediction;
                    const normalizedPrediction = Math.max(0, Math.min(1, predictedValue));

                    results.push({
                        grade: grade,
                        employed: employed,
                        married: married,
                        age: defaultAge,
                        graduationProbability: Number(normalizedPrediction.toFixed(3)),
                        willGraduate: normalizedPrediction >= 0.5
                    });
                }
            }
        }

        // Sort results by grade, then employed status, then married status
        results.sort((a, b) => {
            if (a.grade !== b.grade) return a.grade - b.grade;
            if (a.employed !== b.employed) return a.employed - b.employed;
            return a.married - b.married;
        });

        // Group results for better readability
        const groupedResults = {};
        for (const result of results) {
            const gradeKey = result.grade.toFixed(1);
            if (!groupedResults[gradeKey]) {
                groupedResults[gradeKey] = [];
            }
            groupedResults[gradeKey].push(result);
        }

        // Calculate statistics
        const stats = {
            totalTests: results.length,
            averageGraduationRate: results.reduce((sum, r) => sum + r.graduationProbability, 0) / results.length,
            byEmploymentStatus: {
                employed: results.filter(r => r.employed === 1),
                unemployed: results.filter(r => r.employed === 0)
            },
            byMaritalStatus: {
                married: results.filter(r => r.married === 1),
                unmarried: results.filter(r => r.married === 0)
            }
        };

        // Calculate average graduation rates by different combinations
        stats.averageRates = {
            employed: {
                all: stats.byEmploymentStatus.employed.reduce((sum, r) => sum + r.graduationProbability, 0) / 
                     stats.byEmploymentStatus.employed.length,
                married: stats.byEmploymentStatus.employed.filter(r => r.married === 1)
                    .reduce((sum, r) => sum + r.graduationProbability, 0) / 
                    stats.byEmploymentStatus.employed.filter(r => r.married === 1).length,
                unmarried: stats.byEmploymentStatus.employed.filter(r => r.married === 0)
                    .reduce((sum, r) => sum + r.graduationProbability, 0) / 
                    stats.byEmploymentStatus.employed.filter(r => r.married === 0).length
            },
            unemployed: {
                all: stats.byEmploymentStatus.unemployed.reduce((sum, r) => sum + r.graduationProbability, 0) / 
                     stats.byEmploymentStatus.unemployed.length,
                married: stats.byEmploymentStatus.unemployed.filter(r => r.married === 1)
                    .reduce((sum, r) => sum + r.graduationProbability, 0) / 
                    stats.byEmploymentStatus.unemployed.filter(r => r.married === 1).length,
                unmarried: stats.byEmploymentStatus.unemployed.filter(r => r.married === 0)
                    .reduce((sum, r) => sum + r.graduationProbability, 0) / 
                    stats.byEmploymentStatus.unemployed.filter(r => r.married === 0).length
            }
        };

        return {
            status: 'success',
            results: groupedResults,
            stats: {
                ...stats,
                averageRates: Object.fromEntries(
                    Object.entries(stats.averageRates).map(([key, value]) => [
                        key,
                        Object.fromEntries(
                            Object.entries(value).map(([k, v]) => [k, Number(v.toFixed(3))])
                        )
                    ])
                )
            }
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