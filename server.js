const express = require('express');
const cors = require('cors');
const fs = require('fs');
const csv = require('csv-parser');
const { mean, pow, sqrt } = require('mathjs');
const app = express();
const port = 5000;

app.use(cors());

let studentData = [];

const getSuccessRate = (grade) => {
    // Convert grade to number for comparison
    const targetGrade = Number(grade);
    
    // First try exact match
    const exactMatch = studentData.find(item => Number(item.grade) === targetGrade);
    if (exactMatch) {
        return Number(exactMatch.successRate);
    }

    // Sort data by grade for interpolation
    const sortedData = studentData
        .map(item => ({ 
            grade: Number(item.grade), 
            successRate: Number(item.successRate) 
        }))
        .sort((a, b) => a.grade - b.grade);

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
    if (!lowerGrade) {
        return upperGrade.successRate; // Below lowest grade
    }
    if (!upperGrade) {
        return lowerGrade.successRate; // Above highest grade
    }

    // Linear interpolation
    const gradeDiff = upperGrade.grade - lowerGrade.grade;
    const rateDiff = upperGrade.successRate - lowerGrade.successRate;
    const ratio = (targetGrade - lowerGrade.grade) / gradeDiff;
    
    return lowerGrade.successRate + (rateDiff * ratio);
}

fs.createReadStream('data/grade-data')
    .pipe(csv({
        separator: ' ',
        headers: ['grade', 'successRate']
    }))
    .on('data', (row) => {
        studentData.push({
            grade: row.grade,
            successRate: row.successRate
        });
    });

app.get('/predict', (req, res) => {
    const grade = req.query.grade;
    const successRate = getSuccessRate(grade);
    res.json(successRate);
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});




