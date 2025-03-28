const generatePlotData = (regressionModel, scaleFeatures) => {
    if (!regressionModel) {
        return {
            status: 'error',
            message: 'Model not trained'
        };
    }

    try {
        // Generate points for smooth curves at different ages
        const ages = [20, 25, 30, 35];  // Representative ages
        const points = {
            byAge: {},  // Store curves for each age
            byAgeGPA: {} // Store age curves at fixed GPAs
        };

        // Initialize data structures
        for (const age of ages) {
            points.byAge[age] = {
                employed: {
                    married: [],
                    unmarried: []
                },
                unemployed: {
                    married: [],
                    unmarried: []
                }
            };
        }

        // Fixed GPAs for age variation curves
        const fixedGPAs = [2.5, 3.0, 3.5];
        for (const gpa of fixedGPAs) {
            points.byAgeGPA[gpa] = {
                employed: {
                    married: [],
                    unmarried: []
                },
                unemployed: {
                    married: [],
                    unmarried: []
                }
            };
        }

        // Generate GPA curves for each age
        const numPoints = 50;
        for (const age of ages) {
            for (let i = 0; i < numPoints; i++) {
                const grade = 1.0 + (i * (3.0 / (numPoints - 1)));

                // Test all combinations of employment and marital status
                for (const employed of [0, 1]) {
                    for (const married of [0, 1]) {
                        const features = scaleFeatures(grade, age, married);
                        const model = employed ? regressionModel.models.employed : regressionModel.models.unemployed;

                        if (!model) {
                            console.log('No model available for employment status:', employed);
                            continue;
                        }

                        const prediction = model.predict(features);
                        const predictedValue = Array.isArray(prediction) ? prediction[0] : prediction;
                        const normalizedPrediction = Math.max(0, Math.min(1, predictedValue));

                        const point = {
                            grade: Number(grade.toFixed(2)),
                            probability: Number(normalizedPrediction.toFixed(3))
                        };

                        // Add point to appropriate array based on employment and marital status
                        const empStatus = employed ? 'employed' : 'unemployed';
                        const marStatus = married ? 'married' : 'unmarried';
                        points.byAge[age][empStatus][marStatus].push(point);
                    }
                }
            }
        }

        // Generate age variation curves for each fixed GPA
        const agePoints = 50;
        const ageRange = { min: 18, max: 40 };
        for (let i = 0; i < agePoints; i++) {
            const age = ageRange.min + (i * ((ageRange.max - ageRange.min) / (agePoints - 1)));

            for (const gpa of fixedGPAs) {
                // Test all combinations of employment and marital status
                for (const employed of [0, 1]) {
                    for (const married of [0, 1]) {
                        const features = scaleFeatures(gpa, age, married);
                        const model = employed ? regressionModel.models.employed : regressionModel.models.unemployed;

                        if (!model) continue;

                        const prediction = model.predict(features);
                        const predictedValue = Array.isArray(prediction) ? prediction[0] : prediction;
                        const normalizedPrediction = Math.max(0, Math.min(1, predictedValue));

                        const point = {
                            age: Number(age.toFixed(1)),
                            probability: Number(normalizedPrediction.toFixed(3))
                        };

                        // Add point to appropriate array
                        const empStatus = employed ? 'employed' : 'unemployed';
                        const marStatus = married ? 'married' : 'unmarried';
                        points.byAgeGPA[gpa][empStatus][marStatus].push(point);
                    }
                }
            }
        }

        // Sort all point arrays
        for (const age in points.byAge) {
            for (const empStatus in points.byAge[age]) {
                for (const marStatus in points.byAge[age][empStatus]) {
                    points.byAge[age][empStatus][marStatus].sort((a, b) => a.grade - b.grade);
                }
            }
        }

        for (const gpa in points.byAgeGPA) {
            for (const empStatus in points.byAgeGPA[gpa]) {
                for (const marStatus in points.byAgeGPA[gpa][empStatus]) {
                    points.byAgeGPA[gpa][empStatus][marStatus].sort((a, b) => a.age - b.age);
                }
            }
        }

        // Calculate statistics for each curve
        const stats = {
            byAge: {},
            byAgeGPA: {}
        };

        // Calculate stats for GPA curves at each age
        for (const age of ages) {
            stats.byAge[age] = {
                employed: {
                    married: calculateCurveStats(points.byAge[age].employed.married),
                    unmarried: calculateCurveStats(points.byAge[age].employed.unmarried)
                },
                unemployed: {
                    married: calculateCurveStats(points.byAge[age].unemployed.married),
                    unmarried: calculateCurveStats(points.byAge[age].unemployed.unmarried)
                }
            };
        }

        // Calculate stats for age curves at each GPA
        for (const gpa of fixedGPAs) {
            stats.byAgeGPA[gpa] = {
                employed: {
                    married: calculateCurveStats(points.byAgeGPA[gpa].employed.married),
                    unmarried: calculateCurveStats(points.byAgeGPA[gpa].employed.unmarried)
                },
                unemployed: {
                    married: calculateCurveStats(points.byAgeGPA[gpa].unemployed.married),
                    unmarried: calculateCurveStats(points.byAgeGPA[gpa].unemployed.unmarried)
                }
            };
        }

        // Calculate R-squared values for each demographic group
        const rSquared = {
            employed: {
                married: null,
                unmarried: null
            },
            unemployed: {
                married: null,
                unmarried: null
            }
        };

        // Get actual data points from the model's training data
        const { models } = regressionModel;
        const defaultAge = 20;

        if (models.employed) {
            // Calculate R-squared for employed & married
            const employedMarriedData = regressionModel.stats.modelStats.employed.dataPoints
                .filter(point => point.married === 1)
                .map(point => ({
                    features: scaleFeatures(point.grade, defaultAge, 1),
                    actual: point.graduate
                }));
            rSquared.employed.married = calculateRSquared(models.employed, employedMarriedData);

            // Calculate R-squared for employed & unmarried
            const employedUnmarriedData = regressionModel.stats.modelStats.employed.dataPoints
                .filter(point => point.married === 0)
                .map(point => ({
                    features: scaleFeatures(point.grade, defaultAge, 0),
                    actual: point.graduate
                }));
            rSquared.employed.unmarried = calculateRSquared(models.employed, employedUnmarriedData);
        }

        if (models.unemployed) {
            // Calculate R-squared for unemployed & married
            const unemployedMarriedData = regressionModel.stats.modelStats.unemployed.dataPoints
                .filter(point => point.married === 1)
                .map(point => ({
                    features: scaleFeatures(point.grade, defaultAge, 1),
                    actual: point.graduate
                }));
            rSquared.unemployed.married = calculateRSquared(models.unemployed, unemployedMarriedData);

            // Calculate R-squared for unemployed & unmarried
            const unemployedUnmarriedData = regressionModel.stats.modelStats.unemployed.dataPoints
                .filter(point => point.married === 0)
                .map(point => ({
                    features: scaleFeatures(point.grade, defaultAge, 0),
                    actual: point.graduate
                }));
            rSquared.unemployed.unmarried = calculateRSquared(models.unemployed, unemployedUnmarriedData);
        }

        return {
            status: 'success',
            curves: points.byAge[20], // Default age curves
            ageVariation: {
                ages,
                curves: points.byAge,
                fixedGPAs,
                gpaVariation: points.byAgeGPA
            },
            stats: {
                employed: {
                    married: {
                        rSquared: rSquared.employed.married,
                        ...stats.byAge[20].employed.married
                    },
                    unmarried: {
                        rSquared: rSquared.employed.unmarried,
                        ...stats.byAge[20].employed.unmarried
                    }
                },
                unemployed: {
                    married: {
                        rSquared: rSquared.unemployed.married,
                        ...stats.byAge[20].unemployed.married
                    },
                    unmarried: {
                        rSquared: rSquared.unemployed.unmarried,
                        ...stats.byAge[20].unemployed.unmarried
                    }
                },
                byAge: stats.byAge,
                byAgeGPA: stats.byAgeGPA
            },
            modelType: 'polynomial-regression'
        };
    } catch (error) {
        return {
            status: 'error',
            message: 'Error generating plot data',
            error: error.message
        };
    }
};

// Helper function to calculate statistics for each curve
const calculateCurveStats = (points) => {
    if (!points || points.length === 0) return null;

    const probabilities = points.map(p => p.probability);
    return {
        minProbability: Math.min(...probabilities),
        maxProbability: Math.max(...probabilities),
        averageProbability: Number((probabilities.reduce((a, b) => a + b, 0) / probabilities.length).toFixed(3)),
        crossoverPoint: findCrossoverPoint(points)
    };
};

// Helper function to find the grade/age where probability crosses 0.5
const findCrossoverPoint = (points) => {
    for (let i = 0; i < points.length - 1; i++) {
        if ((points[i].probability < 0.5 && points[i + 1].probability >= 0.5) ||
            (points[i].probability >= 0.5 && points[i + 1].probability < 0.5)) {
            // Linear interpolation to find more precise crossover point
            const x1 = points[i].grade || points[i].age;
            const x2 = points[i + 1].grade || points[i + 1].age;
            const y1 = points[i].probability;
            const y2 = points[i + 1].probability;
            const crossoverPoint = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1);
            return Number(crossoverPoint.toFixed(2));
        }
    }
    return null;
};

// Helper function to calculate R-squared for a specific model and dataset
const calculateRSquared = (model, data) => {
    if (!data || data.length === 0) return null;

    const actualValues = data.map(d => d.actual);
    const mean = actualValues.reduce((a, b) => a + b, 0) / actualValues.length;
    const totalSS = actualValues.reduce((sum, actual) => sum + Math.pow(actual - mean, 2), 0);
    
    const residualSS = data.reduce((sum, d) => {
        const predicted = model.predict(d.features);
        const predictedValue = Array.isArray(predicted) ? predicted[0] : predicted;
        return sum + Math.pow(predictedValue - d.actual, 2);
    }, 0);

    const rSquared = 1 - (residualSS / totalSS);
    return Number(rSquared.toFixed(3));
};

export { generatePlotData }; 