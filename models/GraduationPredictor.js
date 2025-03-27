import { PolynomialRegression } from 'ml-regression-polynomial';

// Configuration
const CONFIG = {
    PORT: 5000,
    MIN_DATA_POINTS: 10,
    MIN_ACCURACY: 0.45,  // Slightly lowered initial threshold to allow model to train
    GRADE_RANGE: { MIN: 0, MAX: 4.0 },
    AGE_RANGE: { MIN: 16, MAX: 100 },
    CV_FOLDS: 5,
    THRESHOLD_RANGE: { MIN: 0.3, MAX: 0.7, STEP: 0.05 },
    TEST_GRADES: [0.0, 1.0, 2.0, 3.0, 4.0],
    TEST_AGES: [18, 25, 35, 45],
    FEATURE_SCALE: 0.1,
    FEATURE_WEIGHTS: {
        GRADE: 0.6,    // Increased grade weight as it's most important
        AGE: 0.15,     // Reduced age impact
        EMPLOYED: 0.15, // Reduced employment impact
        MARRIED: 0.1   // Maintained marital status weight
    }
};

// Type definitions (for documentation)
/**
 * @typedef {Object} ModelWrapper
 * @property {function} predict - Prediction function
 * @property {number} degree - Polynomial degree
 * @property {number} f1Score - F1 score from validation
 * @property {number} threshold - Classification threshold
 * @property {Object} features - Feature normalization info
 */

/**
 * @typedef {Object} ModelStats
 * @property {number} accuracy - Model accuracy
 * @property {number} precision - Model precision
 * @property {number} recall - Model recall
 * @property {number} f1Score - F1 score
 * @property {number} specificity - Model specificity
 * @property {Object} confusionMatrix - Confusion matrix data
 */

class GraduationPredictor {
    constructor() {
        this.model = null;
        this.stats = null;
    }

    validateDataPoint(item) {
        const grade = Number(item.grade);
        const age = Number(item.age);
        const graduate = Number(item.graduate);
        const employed = Number(item.employed);
        const married = Number(item.married);
        
        const isValid = !isNaN(grade) && !isNaN(graduate) && !isNaN(age) &&
                       (employed === 0 || employed === 1) &&
                       (married === 0 || married === 1) &&
                       grade >= 0 && grade <= 100 && 
                       (graduate === 0 || graduate === 1) &&
                       age >= 16 && age <= 100;
                       
        if (!isValid) {
            console.log('Invalid data point:', item, 'Reason:', 
                isNaN(grade) ? 'Invalid grade' :
                isNaN(age) ? 'Invalid age' :
                isNaN(graduate) ? 'Invalid graduate' :
                (employed !== 0 && employed !== 1) ? 'Invalid employed status' :
                (married !== 0 && married !== 1) ? 'Invalid married status' :
                'Values out of range');
        }
        return isValid;
    }

    calculateStats(validData) {
        const gradeStats = {
            min: Math.min(...validData.map(d => Number(d.grade))),
            max: Math.max(...validData.map(d => Number(d.grade))),
            mean: validData.reduce((sum, d) => sum + Number(d.grade), 0) / validData.length,
            stdDev: Math.sqrt(
                validData.reduce((sum, d) => sum + Math.pow(Number(d.grade) - 
                    validData.reduce((s, v) => s + Number(v.grade), 0) / validData.length, 2), 0) / validData.length
            )
        };

        const ageStats = {
            min: Math.min(...validData.map(d => Number(d.age))),
            max: Math.max(...validData.map(d => Number(d.age))),
            mean: validData.reduce((sum, d) => sum + Number(d.age), 0) / validData.length,
            stdDev: Math.sqrt(
                validData.reduce((sum, d) => sum + Math.pow(Number(d.age) - 
                    validData.reduce((s, v) => s + Number(v.age), 0) / validData.length, 2), 0) / validData.length
            )
        };

        return { grade: gradeStats, age: ageStats };
    }

    prepareFeatures(item, stats) {
        console.log('Preparing features for:', item);
        console.log('Using stats:', stats);
        
        const gradeZScore = (Number(item.grade) - stats.grade.mean) / stats.grade.stdDev;
        const ageZScore = (Number(item.age) - stats.age.mean) / stats.age.stdDev;
        const employed = Number(item.employed);
        const married = Number(item.married);

        // Log normalized values
        console.log('Normalized values:', {
            gradeZScore,
            ageZScore,
            employed,
            married
        });
        
        const features = [
            gradeZScore,
            Math.pow(gradeZScore, 2),
            Math.pow(gradeZScore, 3),
            ageZScore,
            Math.pow(ageZScore, 2),
            employed,
            married,
            gradeZScore * employed,
            gradeZScore * ageZScore,
            ageZScore * employed,
            employed * married,
            gradeZScore * married,
            ageZScore * married,
            Number(gradeZScore > 1),
            Number(gradeZScore > 0),
            Number(gradeZScore < -1)
        ];

        console.log('Generated features:', features);
        return features;
    }

    evaluateFold(foldModel, testFeatures, testTargets, threshold) {
        let truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        const predictions = [];
        
        testFeatures.forEach((feature, j) => {
            const rawPrediction = foldModel.predict(feature);
            const predicted = rawPrediction >= threshold ? 1 : 0;
            const actual = testTargets[j];
            
            predictions.push({
                predicted: rawPrediction,
                thresholdedPrediction: predicted,
                actual,
                correct: predicted === actual
            });
            
            if (predicted === 1 && actual === 1) truePos++;
            if (predicted === 1 && actual === 0) falsePos++;
            if (predicted === 0 && actual === 0) trueNeg++;
            if (predicted === 0 && actual === 1) falseNeg++;
        });
        
        const accuracy = (truePos + trueNeg) / testTargets.length;
        const precision = truePos / (truePos + falsePos) || 0;
        const recall = truePos / (truePos + falseNeg) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        // Calculate error analysis
        const errorAnalysis = {
            totalSamples: testTargets.length,
            correctPredictions: truePos + trueNeg,
            incorrectPredictions: falsePos + falseNeg,
            accuracy: accuracy * 100,
            precision: precision * 100,
            recall: recall * 100,
            f1Score: f1 * 100,
            confusionMatrix: {
                truePositives: truePos,
                trueNegatives: trueNeg,
                falsePositives: falsePos,
                falseNegatives: falseNeg
            },
            classBalance: {
                actualPositives: truePos + falseNeg,
                actualNegatives: trueNeg + falsePos,
                predictedPositives: truePos + falsePos,
                predictedNegatives: trueNeg + falseNeg
            }
        };

        return { accuracy, precision, recall, f1, errorAnalysis, predictions };
    }

    // findOptimalThreshold(features, targets, folds = 5) {
    //     const thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7];
    //     let bestThreshold = 0.5;
    //     let bestF1 = 0;
    //     let bestAnalysis = null;

    //     // Calculate positive and negative indices first
    //     const positiveIndices = targets.map((t, i) => t === 1 ? i : -1).filter(i => i !== -1);
    //     const negativeIndices = targets.map((t, i) => t === 0 ? i : -1).filter(i => i !== -1);
        
    //     const positiveFoldSize = Math.floor(positiveIndices.length / folds);
    //     const negativeFoldSize = Math.floor(negativeIndices.length / folds);

    //     console.log('\nStarting cross-validation with', folds, 'folds');
    //     console.log('Class distribution:', {
    //         totalSamples: targets.length,
    //         positives: positiveIndices.length,
    //         negatives: negativeIndices.length,
    //         positiveFoldSize,
    //         negativeFoldSize
    //     });

    //     const thresholdResults = [];

    //     for (const threshold of thresholds) {
    //         let thresholdF1 = 0;
    //         const foldAnalyses = [];
            
    //         for (let i = 0; i < folds; i++) {
    //             const testIndices = [
    //                 ...positiveIndices.slice(i * positiveFoldSize, (i + 1) * positiveFoldSize),
    //                 ...negativeIndices.slice(i * negativeFoldSize, (i + 1) * negativeFoldSize)
    //             ];
                
    //             const trainIndices = Array.from(Array(features.length).keys())
    //                 .filter(idx => !testIndices.includes(idx));
                
    //             const trainFeatures = trainIndices.map(idx => features[idx]);
    //             const trainTargets = trainIndices.map(idx => targets[idx]);
    //             const testFeatures = testIndices.map(idx => features[idx]);
    //             const testTargets = testIndices.map(idx => targets[idx]);
                
    //             const foldModel = new PolynomialRegression(trainFeatures, trainTargets, 1);
    //             const { f1, errorAnalysis } = this.evaluateFold(foldModel, testFeatures, testTargets, threshold);
                
    //             thresholdF1 += f1;
    //             foldAnalyses.push(errorAnalysis);
    //         }
            
    //         const avgF1 = thresholdF1 / folds;
            
    //         // Aggregate analyses across folds
    //         const aggregateAnalysis = foldAnalyses.reduce((agg, curr) => ({
    //             totalSamples: agg.totalSamples + curr.totalSamples,
    //             correctPredictions: agg.correctPredictions + curr.correctPredictions,
    //             incorrectPredictions: agg.incorrectPredictions + curr.incorrectPredictions,
    //             confusionMatrix: {
    //                 truePositives: agg.confusionMatrix.truePositives + curr.confusionMatrix.truePositives,
    //                 trueNegatives: agg.confusionMatrix.trueNegatives + curr.confusionMatrix.trueNegatives,
    //                 falsePositives: agg.confusionMatrix.falsePositives + curr.confusionMatrix.falsePositives,
    //                 falseNegatives: agg.confusionMatrix.falseNegatives + curr.confusionMatrix.falseNegatives
    //             }
    //         }), {
    //             totalSamples: 0,
    //             correctPredictions: 0,
    //             incorrectPredictions: 0,
    //             confusionMatrix: { truePositives: 0, trueNegatives: 0, falsePositives: 0, falseNegatives: 0 }
    //         });

    //         thresholdResults.push({
    //             threshold,
    //             f1Score: avgF1,
    //             analysis: aggregateAnalysis
    //         });
            
    //         if (avgF1 > bestF1) {
    //             bestF1 = avgF1;
    //             bestThreshold = threshold;
    //             bestAnalysis = aggregateAnalysis;
    //         }
    //     }

    //     // Log detailed performance analysis
    //     console.log('\nPerformance analysis across thresholds:');
    //     thresholdResults.forEach(result => {
    //         console.log(`\nThreshold ${result.threshold}:`);
    //         console.log('F1 Score:', (result.f1Score * 100).toFixed(1) + '%');
    //         console.log('Confusion Matrix:', {
    //             truePositives: result.analysis.confusionMatrix.truePositives,
    //             trueNegatives: result.analysis.confusionMatrix.trueNegatives,
    //             falsePositives: result.analysis.confusionMatrix.falsePositives,
    //             falseNegatives: result.analysis.confusionMatrix.falseNegatives
    //         });
    //         console.log('Accuracy:', (result.analysis.correctPredictions / result.analysis.totalSamples * 100).toFixed(1) + '%');
    //     });

    //     console.log('\nBest threshold found:', bestThreshold);
    //     console.log('Best F1 Score:', (bestF1 * 100).toFixed(1) + '%');
    //     console.log('Final Confusion Matrix:', bestAnalysis.confusionMatrix);
    //     console.log('Overall Accuracy:', (bestAnalysis.correctPredictions / bestAnalysis.totalSamples * 100).toFixed(1) + '%');

    //     return { 
    //         threshold: bestThreshold, 
    //         f1Score: bestF1,
    //         analysis: bestAnalysis
    //     };
    // }

    train(data) {
        if (!Array.isArray(data) || data.length < 2) {
            console.log('Not enough data points to train model');
            return false;
        }

        console.log('\nStarting model training...');
        console.log('Total data points:', data.length);

        // Validate data
        const validData = data.filter(item => this.validateDataPoint(item));
        console.log('Valid data points:', validData.length);
        console.log('Invalid data points:', data.length - validData.length);

        if (validData.length < 2) {
            console.log('Not enough valid data points');
            return false;
        }

        try {
            // Calculate statistics
            const stats = this.calculateStats(validData);
            this.stats = stats;  // Store stats for future predictions
            
            // Log data distribution
            const graduateCount = validData.filter(d => Number(d.graduate) === 1).length;
            const employedCount = validData.filter(d => Number(d.employed) === 1).length;
            const marriedCount = validData.filter(d => Number(d.married) === 1).length;

            console.log('Data Statistics:', {
                total: data.length,
                valid: validData.length,
                grade: {
                    min: stats.grade.min,
                    max: stats.grade.max,
                    mean: stats.grade.mean.toFixed(1),
                    stdDev: stats.grade.stdDev.toFixed(1)
                },
                age: {
                    min: stats.age.min,
                    max: stats.age.max,
                    mean: stats.age.mean.toFixed(1),
                    stdDev: stats.age.stdDev.toFixed(1)
                },
                rates: {
                    graduate: ((graduateCount / validData.length) * 100).toFixed(1) + '%',
                    employed: ((employedCount / validData.length) * 100).toFixed(1) + '%',
                    married: ((marriedCount / validData.length) * 100).toFixed(1) + '%'
                }
            });

            // Prepare features and targets
            const features = validData.map(item => this.prepareFeatures(item, stats));
            const targets = validData.map(item => Number(item.graduate));

            // Convert features array to the correct format for PolynomialRegression
            const x = features.map(f => [f[0]]); // Use only the first feature (gradeZScore) as input
            const y = targets;

            // Train final model with degree 2 polynomial
            this.model = new PolynomialRegression(x, y, 2);
            console.log('Model trained successfully');
            return true;

        } catch (error) {
            console.error('Error training model:', error);
            return false;
        }
    }

    predict(grade, employed = 0, age = 25, married = 0) {
        console.log('\nStarting prediction for:', { grade, employed, age, married });
        
        if (!this.model || !this.stats) {
            console.log('Model state:', {
                hasModel: !!this.model,
                hasStats: !!this.stats
            });
            return null;
        }

        try {
            // Validate input values
            if (isNaN(Number(grade)) || isNaN(Number(age))) {
                console.log('Invalid input values:', { grade, age });
                return null;
            }

            const features = this.prepareFeatures({
                grade: Number(grade),
                age: Number(age),
                employed: Number(employed),
                married: Number(married)
            }, this.stats);

            // Use only the first feature (gradeZScore) for prediction
            console.log('Making prediction with feature:', features[0]);
            const prediction = this.model.predict([features[0]]);
            console.log('Raw model prediction:', prediction);
            
            if (isNaN(prediction)) {
                console.log('Invalid prediction value:', prediction);
                return null;
            }
            
            const probability = Math.max(0, Math.min(1, prediction));
            const result = {
                probability: Number((probability * 100).toFixed(1)),
                willGraduate: probability >= 0.5
            };
            
            console.log('Final prediction result:', result);
            return result;
        } catch (error) {
            console.error('Error during prediction:', error);
            return null;
        }
    }
}

export default GraduationPredictor; 