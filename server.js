import express from 'express';
import cors from 'cors';
import fs from 'fs';
import { createRequire } from 'module';
import { generatePlotData } from './plot.js';
import { generateTestData } from './test.js';
import { trainModel } from './train.js';
import { getSuccessRate } from './predict.js';

// Create require for csv-parser which doesn't support ES modules
const require = createRequire(import.meta.url);
const csvParser = require('csv-parser');

const app = express();
const port = 5000;

app.use(cors());

let studentData = [];
let regressionModel = null;
let scaleFeatures = null;  // Will hold the scaling function

fs.createReadStream('data/student-data')
    .pipe(csvParser({
        separator: ' ',
        headers: ['grade', 'employed', 'graduate', 'age', 'married']
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
            graduate: isGraduate ? 1 : 0,
            age: row.age || '20',  // Default age if not provided
            married: row.married === 'true' ? 1 : 0  // Default to unmarried if not provided
        });
    })
    .on('end', () => {
        const result = trainModel(studentData);
        if (result) {
            regressionModel = result.regressionModel;
            scaleFeatures = result.scaleFeatures;
        }
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
        const testResults = generateTestData(regressionModel, scaleFeatures);
        res.json({
            ...testResults,
            dataPoints: studentData.length,
            sampleData: studentData.slice(0, 3).map(item => ({
                grade: Number(item.grade),
                employed: Number(item.employed),
                age: Number(item.age),
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
    const age = req.query.age || 20;  // Default age if not provided
    const married = req.query.married === '1' || req.query.married === 'true' ? 1 : 0;
    const graduationProbability = getSuccessRate(grade, employed, age, false, regressionModel, scaleFeatures, studentData, married);
    
    res.json({
        grade: Number(grade),
        employed: employed,
        age: Number(age),
        married: married,
        graduationProbability: graduationProbability,
        willGraduate: graduationProbability >= 0.5,
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
        const plotData = generatePlotData(regressionModel, scaleFeatures);
        
        // Get actual data points
        const actualPoints = studentData.map(item => ({
            grade: Number(item.grade),
            employed: Number(item.employed),
            graduate: Number(item.graduate),
            age: Number(item.age),
            married: Number(item.married)
        }));

        res.json({
            ...plotData,
            actualPoints
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




