/**
 * Example: Using Forecastly API with JavaScript (Node.js)
 *
 * Install axios first:
 *   npm install axios
 *
 * Run:
 *   node javascript_example.js
 */

const axios = require('axios');

class ForecastlyClient {
    /**
     * Create a new Forecastly API client.
     * @param {string} baseUrl - Base URL of the API
     */
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
            }
        });
    }

    /**
     * Check API health
     * @returns {Promise<Object>}
     */
    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }

    /**
     * Get list of available SKUs
     * @returns {Promise<string[]>}
     */
    async getSKUs() {
        const response = await this.client.get('/api/v1/skus');
        return response.data.skus;
    }

    /**
     * Get forecast for a specific SKU
     * @param {string} skuId - SKU identifier
     * @param {number} horizon - Forecast horizon in days
     * @returns {Promise<Object>}
     */
    async getForecast(skuId, horizon = 14) {
        const response = await this.client.get('/api/v1/predict', {
            params: { sku_id: skuId, horizon: horizon }
        });
        return response.data;
    }

    /**
     * Get model performance metrics
     * @returns {Promise<Object>}
     */
    async getMetrics() {
        const response = await this.client.get('/api/v1/metrics');
        return response.data;
    }

    /**
     * Rebuild forecasts
     * @param {number} horizon - Forecast horizon in days
     * @param {boolean} saveToDb - Save to database
     * @returns {Promise<Object>}
     */
    async rebuildForecasts(horizon = 14, saveToDb = false) {
        const response = await this.client.post('/api/v1/predict/rebuild', null, {
            params: { horizon: horizon, save_to_db: saveToDb }
        });
        return response.data;
    }

    /**
     * Get system status
     * @returns {Promise<Object>}
     */
    async getSystemStatus() {
        const response = await this.client.get('/api/v1/status');
        return response.data;
    }
}

// ==============================================================================
// Usage Examples
// ==============================================================================

async function main() {
    const client = new ForecastlyClient('http://localhost:8000');

    console.log('='.repeat(60));
    console.log('Forecastly API - JavaScript Client Example');
    console.log('='.repeat(60));
    console.log();

    try {
        // 1. Health Check
        console.log('1. Health Check');
        console.log('-'.repeat(60));
        const health = await client.healthCheck();
        console.log(`Status: ${health.status}`);
        console.log(`Service: ${health.service} v${health.version}`);
        console.log();

        // 2. Get Available SKUs
        console.log('2. Available SKUs');
        console.log('-'.repeat(60));
        const skus = await client.getSKUs();
        console.log(`Found ${skus.length} SKUs: ${skus.slice(0, 5).join(', ')}...`);
        console.log();

        // 3. Get Forecast
        console.log('3. Forecast for SKU001');
        console.log('-'.repeat(60));
        const forecast = await client.getForecast('SKU001', 14);
        console.log(`SKU: ${forecast.sku_id}`);
        console.log(`Horizon: ${forecast.horizon} days`);
        console.log(`Records: ${forecast.count}`);
        console.log(`Source: ${forecast.source}`);
        console.log();

        // Show first 3 predictions
        console.log('First 3 predictions:');
        forecast.predictions.slice(0, 3).forEach(pred => {
            console.log(`  ${pred.date}: ${pred.ensemble.toFixed(1)} units (Prophet: ${pred.prophet.toFixed(1)}, XGB: ${pred.xgb.toFixed(1)})`);
        });
        console.log();

        // 4. Get Metrics
        console.log('4. Model Performance Metrics');
        console.log('-'.repeat(60));
        const metrics = await client.getMetrics();
        console.log(`Total SKUs evaluated: ${metrics.count}`);
        console.log();

        // Show first 3 metrics
        console.log('First 3 metrics:');
        metrics.metrics.slice(0, 3).forEach(m => {
            console.log(`  ${m.sku_id}: Ensemble MAPE = ${m.mape_ens.toFixed(1)}% (Best: ${m.best_model})`);
        });
        console.log();

        // Calculate average MAPE
        const avgMape = metrics.metrics.reduce((sum, m) => sum + m.mape_ens, 0) / metrics.metrics.length;
        console.log(`Average Ensemble MAPE: ${avgMape.toFixed(1)}%`);
        console.log();

        // 5. System Status
        console.log('5. System Status');
        console.log('-'.repeat(60));
        const status = await client.getSystemStatus();
        console.log(`System: ${status.system}`);
        console.log(`Database mode: ${status.database_mode}`);

        if (status.data_available) {
            console.log('Data availability:');
            console.log(`  Raw data: ${status.data_available.raw ? '✓' : '✗'}`);
            console.log(`  Processed: ${status.data_available.processed ? '✓' : '✗'}`);
            console.log(`  Predictions: ${status.data_available.predictions ? '✓' : '✗'}`);
            console.log(`  Metrics: ${status.data_available.metrics ? '✓' : '✗'}`);
        }
        console.log();

        // 6. Batch Processing
        console.log('6. Batch Processing (First 5 SKUs)');
        console.log('-'.repeat(60));
        const batchResults = [];

        for (const sku of skus.slice(0, 5)) {
            try {
                const data = await client.getForecast(sku, 7);
                batchResults.push({
                    sku: sku,
                    avgForecast: data.predictions.reduce((sum, p) => sum + p.ensemble, 0) / data.predictions.length
                });
                console.log(`✓ ${sku}: ${batchResults[batchResults.length - 1].avgForecast.toFixed(1)} units/day`);
            } catch (error) {
                console.log(`✗ ${sku}: ${error.message}`);
            }
        }
        console.log();

        console.log('='.repeat(60));
        console.log('Example complete!');
        console.log('='.repeat(60));

    } catch (error) {
        console.error('Error:', error.message);
        if (error.response) {
            console.error('Response:', error.response.data);
        }
        process.exit(1);
    }
}

// Run example
if (require.main === module) {
    main();
}

module.exports = ForecastlyClient;
