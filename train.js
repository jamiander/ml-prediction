// Custom polynomial regression implementation
class CustomPolynomialRegression {
    constructor() {
        this.coefficients = null;
    }

    // Solve normal equations with regularization
    fit(X, y, lambda = 0.01) {
        const m = X.length;
        const n = X[0].length;
        
        // Add regularization term to prevent singular matrices
        const XtX = new Array(n).fill(0).map(() => new Array(n).fill(0));
        const Xty = new Array(n).fill(0);
        
        // Calculate X^T * X and X^T * y
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let k = 0; k < m; k++) {
                    sum += X[k][i] * X[k][j];
                }
                XtX[i][j] = sum + (i === j ? lambda : 0); // Add regularization on diagonal
            }
            
            let sum = 0;
            for (let k = 0; k < m; k++) {
                sum += X[k][i] * y[k];
            }
            Xty[i] = sum;
        }
        
        // Solve using Gaussian elimination
        this.coefficients = this.gaussianElimination(XtX, Xty);
        return this;
    }
    
    // Gaussian elimination with partial pivoting
    gaussianElimination(A, b) {
        const n = A.length;
        const augmented = A.map((row, i) => [...row, b[i]]);
        
        // Forward elimination
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxEl = Math.abs(augmented[i][i]);
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > maxEl) {
                    maxEl = Math.abs(augmented[k][i]);
                    maxRow = k;
                }
            }
            
            // Swap maximum row with current row
            if (maxRow !== i) {
                [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
            }
            
            // Make all rows below this one 0 in current column
            for (let k = i + 1; k < n; k++) {
                const c = -augmented[k][i] / augmented[i][i];
                for (let j = i; j <= n; j++) {
                    if (i === j) {
                        augmented[k][j] = 0;
                    } else {
                        augmented[k][j] += c * augmented[i][j];
                    }
                }
            }
        }
        
        // Back substitution
        const x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n] / augmented[i][i];
            for (let k = i - 1; k >= 0; k--) {
                augmented[k][n] -= augmented[k][i] * x[i];
            }
        }
        
        return x;
    }
    
    predict(features) {
        if (!this.coefficients) {
            throw new Error('Model not trained');
        }
        
        return features.reduce((sum, x, i) => sum + x * this.coefficients[i], 0);
    }
}

const createScaleFeatures = (minGrade, maxGrade, minAge, maxAge) => {
    return (grade, age) => {
        const scaledGrade = (grade - minGrade) / (maxGrade - minGrade);
        const scaledAge = (age - minAge) / (maxAge - minAge);
        
        // Generate polynomial features up to degree 2
        return [
            1, // constant term
            scaledGrade, // linear terms
            scaledAge,
            scaledGrade * scaledGrade, // quadratic terms
            scaledAge * scaledAge,
            scaledGrade * scaledAge // interaction term
        ];
    };
};

// Calculate R-squared for model evaluation
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

const trainModel = (studentData) => {
    if (studentData.length < 2) {
        console.log('Not enough data points to train model');
        return null;
    }

    // Convert strings to numbers and validate data
    const validData = studentData.filter(item => {
        const grade = Number(item.grade);
        const graduate = Number(item.graduate);
        const employed = Number(item.employed);
        const age = Number(item.age);
        const isValid = !isNaN(grade) && !isNaN(graduate) && !isNaN(age) &&
                       (employed === 0 || employed === 1) && age >= 18 && age <= 40 &&
                       grade >= 0 && grade <= 4.0 && (graduate === 0 || graduate === 1);
        if (!isValid) {
            console.log('Invalid data point:', item, 'Reason:', 
                isNaN(grade) ? 'Invalid grade' :
                isNaN(graduate) ? 'Invalid graduate status' :
                isNaN(age) ? 'Invalid age' :
                (employed !== 0 && employed !== 1) ? 'Invalid employed status' :
                'Values out of range');
        }
        return isValid;
    });

    console.log('Total data points:', studentData.length);
    console.log('Valid data points:', validData.length);
    console.log('Sample of valid data:', validData.slice(0, 5));

    if (validData.length < 2) {
        console.log('Not enough valid numeric data points');
        return null;
    }

    try {
        // Calculate min/max for feature scaling
        const minGrade = Math.min(...validData.map(item => Number(item.grade)));
        const maxGrade = Math.max(...validData.map(item => Number(item.grade)));
        const minAge = Math.min(...validData.map(item => Number(item.age)));
        const maxAge = Math.max(...validData.map(item => Number(item.age)));

        // Create the scaling function
        const scaleFeatures = createScaleFeatures(minGrade, maxGrade, minAge, maxAge);

        // Sort data by grade to ensure consistent ordering
        validData.sort((a, b) => Number(a.grade) - Number(b.grade));

        // Split data by employment status
        const employedData = validData.filter(item => Number(item.employed) === 1);
        console.log('Employed data:', employedData.length);
        const unemployedData = validData.filter(item => Number(item.employed) === 0);

        console.log('Data distribution:', {
            employed: employedData.length,
            unemployed: unemployedData.length
        });

        // Train separate models for employed and unemployed
        const models = {
            employed: null,
            unemployed: null,
            scaling: {
                minGrade,
                maxGrade,
                minAge,
                maxAge
            }
        };

        // Train model for employed students
        if (employedData.length >= 2) {
            const empFeatures = employedData.map(item => 
                scaleFeatures(Number(item.grade), Number(item.age))
            );
            const empTarget = employedData.map(item => Number(item.graduate));
            
            console.log('Training employed model with:', {
                numPoints: employedData.length,
                sampleFeatures: empFeatures.slice(0, 3),
                sampleTargets: empTarget.slice(0, 3)
            });

            models.employed = new CustomPolynomialRegression();
            models.employed.fit(empFeatures, empTarget);
            
            const empCoefficients = models.employed.coefficients;
            console.log('Employed model coefficients:', {
                constant: empCoefficients[0],
                grade: empCoefficients[1],
                age: empCoefficients[2],
                gradeSquared: empCoefficients[3],
                ageSquared: empCoefficients[4],
                gradeAge: empCoefficients[5]
            });
            
            console.log('Employed model equation:');
            console.log(`P(graduate) = ${empCoefficients[0].toFixed(3)} + ` +
                       `${empCoefficients[1].toFixed(3)}*grade + ` +
                       `${empCoefficients[2].toFixed(3)}*age + ` +
                       `${empCoefficients[3].toFixed(3)}*grade² + ` +
                       `${empCoefficients[4].toFixed(3)}*age² + ` +
                       `${empCoefficients[5].toFixed(3)}*grade*age`);
        }

        // Train model for unemployed students
        if (unemployedData.length >= 2) {
            const unempFeatures = unemployedData.map(item => 
                scaleFeatures(Number(item.grade), Number(item.age))
            );
            const unempTarget = unemployedData.map(item => Number(item.graduate));
            
            console.log('Training unemployed model with:', {
                numPoints: unemployedData.length,
                sampleFeatures: unempFeatures.slice(0, 3),
                sampleTargets: unempTarget.slice(0, 3)
            });

            models.unemployed = new CustomPolynomialRegression();
            models.unemployed.fit(unempFeatures, unempTarget);
            
            const unempCoefficients = models.unemployed.coefficients;
            console.log('Unemployed model coefficients:', {
                constant: unempCoefficients[0],
                grade: unempCoefficients[1],
                age: unempCoefficients[2],
                gradeSquared: unempCoefficients[3],
                ageSquared: unempCoefficients[4],
                gradeAge: unempCoefficients[5]
            });
            
            console.log('Unemployed model equation:');
            console.log(`P(graduate) = ${unempCoefficients[0].toFixed(3)} + ` +
                       `${unempCoefficients[1].toFixed(3)}*grade + ` +
                       `${unempCoefficients[2].toFixed(3)}*age + ` +
                       `${unempCoefficients[3].toFixed(3)}*grade² + ` +
                       `${unempCoefficients[4].toFixed(3)}*age² + ` +
                       `${unempCoefficients[5].toFixed(3)}*grade*age`);
        }

        // Test predictions
        let validPredictions = 0;
        let totalError = 0;
        const testResults = [];

        for (const item of validData) {
            const grade = Number(item.grade);
            const age = Number(item.age);
            const employed = Number(item.employed);
            const actual = Number(item.graduate);
            
            try {
                const model = employed ? models.employed : models.unemployed;
                if (!model) continue;

                const features = scaleFeatures(grade, age);
                const predicted = model.predict(features);
                console.log('Features for prediction:', features);
                console.log('Predicted:', predicted);
                const predictedValue = Array.isArray(predicted) ? predicted[0] : predicted;
                
                if (!isNaN(predictedValue)) {
                    const error = Math.abs(predictedValue - actual);
                    validPredictions++;
                    totalError += error;
                    
                    testResults.push({
                        grade,
                        age,
                        employed,
                        actual,
                        predicted: Number(predictedValue.toFixed(3)),
                        error: Number(error.toFixed(3))
                    });
                } else {
                    console.log('Got NaN prediction for:', { grade, age, employed });
                }
            } catch (err) {
                console.error('Prediction error for:', { grade, age, employed }, err.message);
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
            return null;
        }

        // Calculate R-squared for each model
        const modelStats = {
            employed: null,
            unemployed: null
        };

        if (models.employed) {
            const empFeatures = employedData.map(item => 
                scaleFeatures(Number(item.grade), Number(item.age))
            );
            const empTarget = employedData.map(item => Number(item.graduate));
            const empRSquared = calculateRSquared(models.employed, empFeatures, empTarget);
            modelStats.employed = {
                rSquared: Number(empRSquared.toFixed(3)),
                dataPoints: employedData.length
            };
        }

        if (models.unemployed) {
            const unempFeatures = unemployedData.map(item => 
                scaleFeatures(Number(item.grade), Number(item.age))
            );
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
            const result = { 
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
            return { regressionModel: result, scaleFeatures };
        } else {
            console.log('Model predictions unreliable (average error too high), falling back to interpolation');
            return null;
        }
        
    } catch (error) {
        console.error('Error training model:', error);
        return null;
    }
};

export { trainModel, createScaleFeatures, CustomPolynomialRegression }; 