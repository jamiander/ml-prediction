import express from 'express';
import cors from 'cors';
import fs from 'fs';
import { mean, pow, sqrt } from 'mathjs';
import { PolynomialRegression } from 'ml-regression-polynomial';
import { createRequire } from 'module';

// Create require for csv-parser which doesn't support ES modules
const require = createRequire(import.meta.url);
const csvParser = require('csv-parser');

const app = express();
const port = 5000;

app.use(cors());

let studentData = [];
let regressionModel = null;

const trainModel = () => {
    if (studentData.length < 2) {
        console.log('Not enough data points to train model');
        return;
    }

    // Convert strings to numbers and validate data
    const validData = studentData.filter(item => {
        const grade = Number(item.grade);
        const graduate = Number(item.graduate);
        const employed = Number(item.employed);
        const isValid = !isNaN(grade) && !isNaN(graduate) && (employed === 0 || employed === 1) &&
                       grade >= 0 && grade <= 100 && (graduate === 0 || graduate === 1);
        if (!isValid) {
            console.log('Invalid data point:', item, 'Reason:', 
                isNaN(grade) ? 'Invalid grade' :
                isNaN(graduate) ? 'Invalid graduate status' :
                (employed !== 0 && employed === 1) ? 'Invalid employed status' :
                'Values out of range');
        }
        return isValid;
    });

    console.log('Total data points:', studentData.length);
    console.log('Valid data points:', validData.length);
    console.log('Sample of valid data:', validData.slice(0, 5));

    if (validData.length < 2) {
        console.log('Not enough valid numeric data points');
        return;
    }

    try {
        // Sort data by grade to ensure consistent ordering
        validData.sort((a, b) => Number(a.grade) - Number(b.grade));

        // Split data by employment status
        const employedData = validData.filter(item => Number(item.employed) === 1);
        const unemployedData = validData.filter(item => Number(item.employed) === 0);

        console.log('Data distribution:', {
            employed: employedData.length,
            unemployed: unemployedData.length
        });

        // Train separate models for employed and unemployed
        const models = {
            employed: null,
            unemployed: null
        };

        // Train model for employed students
        if (employedData.length >= 2) {
            const empFeatures = employedData.map(item => [Number(item.grade)]);
            const empTarget = employedData.map(item => Number(item.graduate));
            models.employed = new PolynomialRegression(empFeatures, empTarget, 2);
            console.log('Employed model trained with', employedData.length, 'points');
        }

        // Train model for unemployed students
        if (unemployedData.length >= 2) {
            const unempFeatures = unemployedData.map(item => [Number(item.grade)]);
            const unempTarget = unemployedData.map(item => Number(item.graduate));
            models.unemployed = new PolynomialRegression(unempFeatures, unempTarget, 2);
            console.log('Unemployed model trained with', unemployedData.length, 'points');
        }

        // Test predictions
        let validPredictions = 0;
        let totalError = 0;
        const testResults = [];

        for (const item of validData) {
            const grade = Number(item.grade);
            const employed = Number(item.employed);
            const actual = Number(item.graduate);
            
            try {
                const model = employed ? models.employed : models.unemployed;
                if (!model) continue;

                const predicted = model.predict([grade]);
                const predictedValue = Array.isArray(predicted) ? predicted[0] : predicted;
                
                if (!isNaN(predictedValue)) {
                    const error = Math.abs(predictedValue - actual);
                    validPredictions++;
                    totalError += error;
                    
                    testResults.push({
                        grade,
                        employed,
                        actual,
                        predicted: Number(predictedValue.toFixed(1)),
                        error: Number(error.toFixed(1))
                    });
                } else {
                    console.log('Got NaN prediction for:', { grade, employed });
                }
            } catch (err) {
                console.error('Prediction error for:', { grade, employed }, err.message);
            }
        }

        console.log('Prediction test results:', {
            totalTests: validData.length,
            validPredictions,
            averageError: validPredictions > 0 ? (totalError / validPredictions).toFixed(1) : 'N/A',
            samplePredictions: testResults.slice(0, 5)
        });

        if (validPredictions < validData.length * 0.5) {
            console.log('Too many failed predictions, falling back to interpolation');
            regressionModel = null;
            return;
        }

        // After model training, calculate R-squared
        const calculateRSquared = (model, features, actualValues) => {
            // Calculate mean of actual values
            const mean = actualValues.reduce((a, b) => a + b, 0) / actualValues.length;
            
            // Calculate total sum of squares
            const totalSS = actualValues.reduce((sum, actual) => sum + Math.pow(actual - mean, 2), 0);
            
            // Calculate residual sum of squares
            const residualSS = features.reduce((sum, feature, i) => {
                const predicted = model.predict(feature);
                const predictedValue = Array.isArray(predicted) ? predicted[0] : predicted;
                return sum + Math.pow(predictedValue - actualValues[i], 2);
            }, 0);
            
            // Calculate R-squared
            return 1 - (residualSS / totalSS);
        };

        // Calculate R-squared for each model
        const modelStats = {
            employed: null,
            unemployed: null
        };

        if (models.employed) {
            const empFeatures = employedData.map(item => [Number(item.grade)]);
            const empTarget = employedData.map(item => Number(item.graduate));
            const empRSquared = calculateRSquared(models.employed, empFeatures, empTarget);
            modelStats.employed = {
                rSquared: Number(empRSquared.toFixed(3)),
                dataPoints: employedData.length
            };
        }

        if (models.unemployed) {
            const unempFeatures = unemployedData.map(item => [Number(item.grade)]);
            const unempTarget = unemployedData.map(item => Number(item.graduate));
            const unempRSquared = calculateRSquared(models.unemployed, unempFeatures, unempTarget);
            modelStats.unemployed = {
                rSquared: Number(unempRSquared.toFixed(3)),
                dataPoints: unemployedData.length
            };
        }

        console.log('Model Statistics:', {
            employed: modelStats.employed,
            unemployed: modelStats.unemployed
        });

        const avgError = totalError / validPredictions;
        if (!isNaN(avgError) && avgError < 0.5) {  // More lenient threshold to allow polynomial fit
            regressionModel = { 
                models,
                stats: {
                    validPredictions,
                    averageError: avgError,
                    totalDataPoints: validData.length,
                    employedDataPoints: employedData.length,
                    unemployedDataPoints: unemployedData.length,
                    modelStats
                }
            };
            console.log('Models trained successfully');
        } else {
            console.log('Model predictions unreliable (average error too high), falling back to interpolation');
            regressionModel = null;
        }
        
    } catch (error) {
        console.error('Error training model:', error);
        regressionModel = null;
    }
}

const getSuccessRate = (grade, employed = 0, useInterpolation = false) => {
    const targetGrade = Number(grade);
    const isEmployed = Number(employed);
    
    if (isNaN(targetGrade)) {
        console.error('Invalid grade input:', grade);
        return null;
    }
    
    // If model isn't trained yet or we're forced to use interpolation
    if (!regressionModel || useInterpolation) {
        console.log('Using interpolation method');
        // First try exact match
        const exactMatch = studentData.find(item => 
            Number(item.grade) === targetGrade && 
            Number(item.employed) === isEmployed
        );
        if (exactMatch) {
            console.log('Exact match found:', exactMatch);
            return Number(exactMatch.graduate);  // Already 0 or 1
        }

        // Sort data by grade for interpolation (matching employed status)
        const sortedData = studentData
            .filter(item => Number(item.employed) === isEmployed)
            .map(item => ({ 
                grade: Number(item.grade), 
                graduate: Number(item.graduate)  // Already 0 or 1
            }))
            .sort((a, b) => a.grade - b.grade);

        if (sortedData.length === 0) {
            console.log('No data available for this employment status, using all data');
            // If no data for this employment status, use all data
            sortedData.push(...studentData
                .map(item => ({ 
                    grade: Number(item.grade), 
                    graduate: Number(item.graduate)  // Already 0 or 1
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
            return getSuccessRate(grade, employed, true);
        }

        console.log('Making prediction with input:', [targetGrade], 'for employed:', isEmployed);
        
        const prediction = model.predict([targetGrade]);
        const predictedValue = Array.isArray(prediction) ? prediction[0] : prediction;
        console.log('Raw prediction:', predictedValue);
        
        if (isNaN(predictedValue)) {
            console.log('Invalid prediction from model, using interpolation');
            return getSuccessRate(grade, employed, true);
        }
        
        // Normalize prediction between 0 and 1
        const normalizedPrediction = Number((Math.max(0, Math.min(1, predictedValue))).toFixed(3));
        console.log('Final prediction:', normalizedPrediction);
        
        return normalizedPrediction;
    } catch (error) {
        console.error('Error making prediction:', error.message);
        return getSuccessRate(grade, employed, true);
    }
}

fs.createReadStream('data/grade-data')
    .pipe(csvParser({
        separator: ' ',
        headers: ['grade', 'employed', 'graduate']
    }))
    .on('data', (row) => {
        // Convert the third column to a boolean graduate status
        // If it's already true/false, use that, otherwise convert from number (≥ 50 means graduated)
        const graduateValue = row.graduate.toLowerCase();
        const isGraduate = graduateValue === 'true' ? true :
                          graduateValue === 'false' ? false :
                          Number(row.graduate) >= 50;

        studentData.push({
            grade: row.grade,
            employed: row.employed === 'true' ? 1 : 0,
            graduate: isGraduate ? 1 : 0
        });
    })
    .on('end', () => {
        trainModel();
        console.log('Data loaded and model trained');
    });

app.get('/data', (req, res) => {
    res.json({
        rawData: studentData,
        modelTrained: regressionModel !== null,
        dataPoints: studentData.length
    });
});

app.get('/test', (req, res) => {
    if (!regressionModel) {
        return res.json({
            status: 'error',
            message: 'Model not trained or using interpolation',
            usingInterpolation: true
        });
    }

    try {
        // Test predictions on a range of grades with both employed states
        const testGrades = [1, 3, 3, 3.75, 4];
        const results = testGrades.flatMap(grade => {
            try {
                const results = [];
                
                if (regressionModel.models.unemployed) {
                    const unemployedPred = regressionModel.models.unemployed.predict([grade]);
                    const unemployedValue = Array.isArray(unemployedPred) ? unemployedPred[0] : unemployedPred;
                    results.push({
                        grade,
                        employed: 0,
                        prediction: Number(unemployedValue.toFixed(1))
                    });
                }
                
                if (regressionModel.models.employed) {
                    const employedPred = regressionModel.models.employed.predict([grade]);
                    const employedValue = Array.isArray(employedPred) ? employedPred[0] : employedPred;
                    results.push({
                        grade,
                        employed: 1,
                        prediction: Number(employedValue.toFixed(1))
                    });
                }
                
                return results;
            } catch (err) {
                console.error('Error predicting for grade:', grade, err);
                return [];
            }
        });

        res.json({
            status: 'success',
            modelTrained: true,
            testResults: results,
            modelStats: regressionModel.stats.modelStats,
            dataPoints: studentData.length,
            sampleData: studentData.slice(0, 3).map(item => ({
                grade: Number(item.grade),
                employed: Number(item.employed),
                graduate: Number(item.graduate)
            }))
        });
    } catch (error) {
        res.json({
            status: 'error',
            message: 'Error testing model',
            error: error.message
        });
    }
});

app.get('/predict', (req, res) => {
    const grade = req.query.grade;
    const employed = req.query.employed === '1' || req.query.employed === 'true' ? 1 : 0;
    const graduationProbability = getSuccessRate(grade, employed);
    
    res.json({
        grade: Number(grade),
        employed: employed,
        graduationProbability: graduationProbability,
        willGraduate: graduationProbability >= 0.5,  // Updated threshold to 0.5 for probability
        modelType: regressionModel ? 'polynomial-regression' : 'linear-interpolation',
        dataPoints: studentData.length
    });
});

app.get('/plot', (req, res) => {
    if (!regressionModel) {
        return res.json({
            status: 'error',
            message: 'Model not trained'
        });
    }

    try {
        // Generate points for smooth curves
        const plotPoints = [];
        for (let grade = 1.0; grade <= 4.0; grade += 0.1) {
            // Get predictions for both employed and unemployed
            let employedPred = null;
            let unemployedPred = null;

            if (regressionModel.models.employed) {
                const pred = regressionModel.models.employed.predict([grade]);
                employedPred = Number(Math.max(0, Math.min(1, Array.isArray(pred) ? pred[0] : pred)).toFixed(3));
            }

            if (regressionModel.models.unemployed) {
                const pred = regressionModel.models.unemployed.predict([grade]);
                unemployedPred = Number(Math.max(0, Math.min(1, Array.isArray(pred) ? pred[0] : pred)).toFixed(3));
            }

            plotPoints.push({
                grade: Number(grade.toFixed(1)),
                employed: employedPred,
                unemployed: unemployedPred
            });
        }

        // Get actual data points
        const actualPoints = studentData.map(item => ({
            grade: Number(item.grade),
            employed: Number(item.employed),
            graduate: Number(item.graduate)
        }));

        // Get model statistics
        const modelStats = regressionModel.stats.modelStats;

        res.json({
            status: 'success',
            plotPoints,
            actualPoints,
            modelStats
        });
    } catch (error) {
        res.json({
            status: 'error',
            message: 'Error generating plot data',
            error: error.message
        });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});




