#include <bits/stdc++.h>
using namespace std;
using cd = complex<double>;

const double pi = acos(-1);
const int COEF_FRAC = 30;

//метод ресемплов Фурье
bool isPowerOfTwo(int n){
    return n>0&&(n&(n - 1))== 0;
}
int nextPowerOfTwo(int n){
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

void fft(vector<cd>& a, bool invert){
    int n = static_cast<int>(a.size());
    int k = 0;
    while ((1 << k) < n)
        k++;
    vector<int> rev(n);
    rev[0] = 0;
    int high1 = -1;

    for (int i = 1; i < n; i++){
        if ((i & (i - 1)) == 0)
            high1++;
        rev[i] = rev[i ^ (1 << high1)];
        rev[i] |= (1 << (k - high1 - 1));
    }

    vector<cd> roots(n);
    for (int i = 0; i < n; i++){
        double alpha = 2.0 * pi * i / n * (invert ? -1.0 : 1.0);
        roots[i] = cd(cos(alpha), sin(alpha));
    }

    vector<cd> cur(n);
    for (int i = 0; i < n; i++)
        cur[i] = a[rev[i]];

    for (int len = 1; len < n; len <<= 1){
        vector<cd> ncur(n);
        int rstep = static_cast<int>(roots.size()) / (len * 2);
        for (int pdest = 0; pdest < n;) {
            int p1 = pdest;
            for (int i = 0; i < len; i++){
                cd val = roots[i * rstep] * cur[p1 + len];
                ncur[pdest] = cur[p1] + val;
                ncur[pdest + len] = cur[p1] - val;
                pdest++;
                p1++;
            }
            pdest += len;
        }
        cur.swap(ncur);
    }

    a.swap(cur);
    if (invert){
        for (cd& x : a)
            x /= n;
    }
}

void arbitraryFft(vector<cd>& a, bool invert){
    int n = static_cast<int>(a.size());
    if (isPowerOfTwo(n)){
        fft(a, invert);
        return;
    }
    int m = nextPowerOfTwo(2*n - 1);
    vector<cd> p(m);
    vector<cd> q(m);
    double sign = invert ? -1.0 : 1.0;
    for (int i = 0; i < n; i++){
        double angle = sign * pi * static_cast<double>(i) * i / n;
        cd w(cos(angle), sin(angle));
        p[i] = a[i] * w;
    }
    for (int i = 0; i < n; i++) {
        double angle = -sign * pi * static_cast<double>(i) * i / n;
        cd w(cos(angle), sin(angle));
        q[i] = w;
        if (i != 0)
            q[m - i] = w;
    }
    fft(p, false);
    fft(q, false);
    for (int i = 0; i < m; i++)
        p[i] *= q[i];
    fft(p, true);
    vector<cd> result(n);
    for (int i = 0; i < n; i++) {
        double angle = sign * pi * static_cast<double>(i) * i / n;
        cd w(cos(angle), sin(angle));

        result[i] = p[i] * w;
    }
    if (invert){
        for (cd& x : result)
            x /= n;
    }
    a.swap(result);
}

vector<double> interpolateFourier(const vector<double>& x, int up){
    int n = static_cast<int>(x.size());
    int m = n * up;
    vector<cd> x_spectrum(n);
    for (int i = 0; i < n; i++)
        x_spectrum[i] = cd(x[i], 0.0);
    arbitraryFft(x_spectrum, false);
    vector<cd> y_spectrum(m, cd(0.0, 0.0));
    y_spectrum[0] = x_spectrum[0];
    if (n % 2 == 0){
        int half = n/2;
        for (int k = 1; k < half; ++k)
            y_spectrum[k] = x_spectrum[k];
        y_spectrum[half] = x_spectrum[half] * 0.5;
        y_spectrum[m - half] = x_spectrum[half] * 0.5;
        for (int k = half + 1; k < n; k++) {
            int neg = n - k;
            y_spectrum[m - neg] = x_spectrum[k];
        }
    }else{
        int half = n/2;
        for (int k = 1; k <= half; k++)
            y_spectrum[k] = x_spectrum[k];
        for (int k = half+1; k < n; k++){
            int neg = n-k;
            y_spectrum[m-neg] = x_spectrum[k];
        }
    }

    arbitraryFft(y_spectrum, true);

    vector<double> y(m);
    double scale = static_cast<double>(m) / n;

    for (int i = 0; i < m; i++)
        y[i] = real(y_spectrum[i])*scale;

    return y;
}
//конец Фурье



//полифазный Кайзер
double normsincfun(double x){
    if (abs(x)<1e-14)
        return 1.0;
    return sin(pi*x)/(pi*x);
}

double besselI0(double x){
    double sum = 1.0;
    double term = 1.0;
    double y = x*x/4.0;
    for (int k = 1; k <= 60; k++){
        term *= y/(static_cast<double>(k) * k);
        sum += term;
        if (abs(term) < 1e-15*abs(sum))
            break;
    }
    return sum;
}

double kaiserWindow(int n, int taps, double beta){
    if (taps == 1)
        return 1.0;
    double a = 2.0*n/ static_cast<double>(taps-1)-1.0;
    double arg = beta*sqrt(max(0.0, 1.0 - a*a));
    return besselI0(arg)/besselI0(beta);
}

vector<double> designKaiserInterpolatorFir(int up, int taps, double beta){
    if (taps <= 0 || taps % 2 == 0)
        throw invalid_argument("taps положительны и нечётны для наличия центрального элемента");

    vector<double> h(taps);
    int mid = taps/2;
    double fc = 1.0/(2.0 * up);
    for (int i = 0; i < taps; i++){
        double t = static_cast<double>(i-mid);
        h[i] = 2.0 *fc* normsincfun(2.0*fc*t)*kaiserWindow(i, taps, beta);
    }
    double sum = accumulate(h.begin(), h.end(), 0.0);
    for (double& v : h)
        v*=static_cast<double>(up)/sum;
    return h;
}

vector<double> interpolateKaiserPolyphase(const vector<double>& x, int up, int taps, double beta){
    vector<double> h = designKaiserInterpolatorFir(up, taps, beta);
    int input_size = static_cast<int>(x.size());
    int output_size = input_size*up;
    int mid = static_cast<int>(h.size()/2);
    vector<vector<double>> phases(up);

    for (int phase = 0; phase < up; phase++){
        for (int k = phase; k < static_cast<int>(h.size()); k+=up)
            phases[phase].push_back(h[k]);
    }

    vector<double> y(output_size, 0.0);
    for (int n = 0; n < output_size; n++){
        int phase = (n+mid)%up;
        int base = (n+mid-phase)/up;
        double acc = 0.0;
        const vector<double>& e = phases[phase];
        for (int p = 0; p < static_cast<int>(e.size()); p++){
            int src = base-p;
            if (0 <= src && src < input_size)
                acc += e[p]*x[src];
        }
        y[n] = acc;
    }

    return y;
}
//конец Кайзера



vector <double> genSin(double samplingRate, double frequency, double dur){
    int sampleCount = static_cast<int>(dur*samplingRate);
    vector <double> signal(sampleCount);
    for(int i=0; i<sampleCount; i++){
        signal[i]=sin(2*pi*frequency*i/samplingRate);
    }
    return signal;
}

int16_t quantizeQ15(double x){
    long long q = llround(x * 32768.0);
    q = clamp(q, -32768LL, 32767LL);
    return static_cast<int16_t>(q);
}

double dequantizeQ15(int16_t q){
    return static_cast<double>(q)/32768.0;
}

vector<int16_t> quantizeVectorQ15(const vector<double>& x){
    vector<int16_t> q(x.size());
    for (size_t i = 0; i < x.size(); i++)
        q[i] = quantizeQ15(x[i]);
    return q;
}

vector<double> dequantizeVectorQ15(const vector<int16_t>& q){
    vector<double> x(q.size());
    for (size_t i = 0; i < q.size(); i++)
        x[i] = dequantizeQ15(q[i]);
    return x;
}


struct QuantizationErrorQ15{
    double mean_error = 0.0;
    double mse = 0.0;
    double max_abs_error = 0.0;
};

QuantizationErrorQ15 computeQuantizationErrorQ15(const vector<double>& original, const vector<double>& restored){
    QuantizationErrorQ15 result;
    for (size_t i = 0; i < original.size(); i++) {
        double error = restored[i] - original[i];

        result.mean_error += error;
        result.mse += error * error;
        result.max_abs_error = max(result.max_abs_error, abs(error));
    }
    double n = static_cast<double>(original.size());
    result.mean_error /= n;
    result.mse /= n;

    return result;
}


struct InterpolationQuality {
    double f_hz = 0.0;
    double mse = 0.0;
    double max_abs_error = 0.0;
    double amp_error = 0.0;
};

double spectrumAmplitudeAt(const vector<double>& y, double fs, double f){
    int n = static_cast<int>(y.size());
    vector<cd> spectrum(n);
    for (int i = 0; i < n; i++)
        spectrum[i] = cd(y[i], 0.0);
    arbitraryFft(spectrum, false);
    int k = static_cast<int>(llround(f*n/fs));
    if (k < 0 || k > n/2)
        return 0.0;
    if (k == 0 || (n%2 == 0 && k == n/2))
        return abs(spectrum[k]) / n;
    return 2.0 * abs(spectrum[k]) / n;
}

InterpolationQuality analyzeInterpolationQuality(const vector<double>& y, double frequency, double samplingRateOut,double dur,int guard){
    vector<double> ideal = genSin(samplingRateOut, frequency, dur);
    int n = min(static_cast<int>(y.size()), static_cast<int>(ideal.size()));
    int begin = guard;
    int end = n-guard;
    if (begin >= end)
        throw invalid_argument("guard больше половины");

    InterpolationQuality q;
    q.f_hz = frequency;
    for (int i = begin; i < end; i++){
        double error = y[i] - ideal[i];
        q.mse += error * error;
        q.max_abs_error = max(q.max_abs_error, abs(error));
    }
    q.mse /= static_cast<double>(end-begin);
    double amp = spectrumAmplitudeAt(y, samplingRateOut, frequency);
    double expected_amp = (frequency == 0.0 ? 0.0 : 1.0);
    q.amp_error = amp-expected_amp;

    return q;
}

long long roundShift(long long x, int shift) {
    long long half = 1LL <<(shift-1);
    if (x >= 0)
        return (x + half) >> shift;
    return -((-x + half) >> shift);
}
int16_t clampQ15Code(long long x) {
    x = clamp(x, -32768LL, 32767LL);
    return static_cast<int16_t>(x);
}

vector<int32_t> quantizeFirCoefficientsQ30(const vector<double>& h){
    vector<int32_t> q(h.size());
    double scale = static_cast<double>(1LL << COEF_FRAC);
    for (size_t i = 0; i < h.size(); i++){
        long long v = llround(h[i] * scale);
        v = clamp(v,static_cast<long long>(numeric_limits<int32_t>::min()),static_cast<long long>(numeric_limits<int32_t>::max()));
        q[i] = static_cast<int32_t>(v);
    }
    return q;
}

vector<int16_t> interpolateFirFixedPolyphaseQ15(const vector<int16_t>& x, const vector<int32_t>& h, int up){
    int inputSize = static_cast<int>(x.size());
    int outputSize = inputSize * up;
    int taps = static_cast<int>(h.size());
    int mid = taps/2;
    vector<vector<int32_t>> phases(up);
    for (int phase = 0; phase < up; phase++){
        for (int k = phase; k < taps; k += up)
            phases[phase].push_back(h[k]);
    }
    vector<int16_t> y(outputSize);
    for (int n = 0; n < outputSize; n++){
        int phase = (n + mid) % up;
        int base = (n + mid - phase) / up;
        long long acc = 0;
        const vector<int32_t>& e = phases[phase];
        for (int p = 0; p < static_cast<int>(e.size()); p++){
            int src = base - p;
            if (0 <= src && src < inputSize)
                acc += static_cast<long long>(x[src]) * e[p];
        }
        long long q15 = roundShift(acc, COEF_FRAC);
        y[n] = clampQ15Code(q15);
    }
    return y;
}

//вывод
struct ExperimentRow {
    double f_hz = 0.0;
    QuantizationErrorQ15 quant;
    InterpolationQuality kaiser;
    InterpolationQuality fourier;
    InterpolationQuality fixed;
};

ExperimentRow evaluateFrequency(double frequency,double samplingRate,double dur,int up,double samplingRateOut,int taps,double beta,int guardKaiser,int guardFourier,const vector<int32_t>& hFixed){
    ExperimentRow row;
    row.f_hz = frequency;
    vector<double> x = genSin(samplingRate, frequency, dur);
    vector<int16_t> xQ15 = quantizeVectorQ15(x);
    vector<double> xRestored = dequantizeVectorQ15(xQ15);
    row.quant = computeQuantizationErrorQ15(x, xRestored);
    vector<double> yKaiser = interpolateKaiserPolyphase(x, up, taps, beta);
    vector<double> yFourier = interpolateFourier(x, up);
    vector<int16_t> yFixedQ15 = interpolateFirFixedPolyphaseQ15(xQ15, hFixed, up);
    vector<double> yFixed = dequantizeVectorQ15(yFixedQ15);
    row.kaiser = analyzeInterpolationQuality(yKaiser,frequency,samplingRateOut,dur,guardKaiser);
    row.fourier = analyzeInterpolationQuality(yFourier,frequency,samplingRateOut,dur,guardFourier);
    row.fixed = analyzeInterpolationQuality(yFixed,frequency,samplingRateOut,dur,guardKaiser);
    return row;
}

void writeFullCsv(const vector<ExperimentRow>& rows, const string& path){
    ofstream out(path);
    out << scientific << setprecision(10);

    out << "f_hz,"
        << "quant_mean_error,quant_mse,quant_max_abs_error,"
        << "kaiser_mse,kaiser_max_abs_error,kaiser_amp_error,"
        << "fourier_mse,fourier_max_abs_error,fourier_amp_error,"
        << "fixed_mse,fixed_max_abs_error,fixed_amp_error,"
        << "fixed_minus_kaiser_mse,fixed_minus_kaiser_amp_error"
        << '\n';

    for (const ExperimentRow& r : rows){
        out << r.f_hz << ','
            << r.quant.mean_error << ','
            << r.quant.mse << ','
            << r.quant.max_abs_error << ','
            << r.kaiser.mse << ','
            << r.kaiser.max_abs_error << ','
            << r.kaiser.amp_error << ','
            << r.fourier.mse << ','
            << r.fourier.max_abs_error << ','
            << r.fourier.amp_error << ','
            << r.fixed.mse << ','
            << r.fixed.max_abs_error << ','
            << r.fixed.amp_error << ','
            << r.fixed.mse - r.kaiser.mse << ','
            << r.fixed.amp_error - r.kaiser.amp_error
            <<'\n';
    }
}

int main(){
    const double samplingRate = 100.0;
    const double dur = 10.0;
    const int up = 2;
    const double samplingRateOut = samplingRate*up;
    const int taps = 257;
    const double beta = 10.0;
    const int guardKaiser = taps/2;
    const int guardFourier = 0;
    vector<double> hDouble = designKaiserInterpolatorFir(up, taps, beta);
    vector<int32_t> hFixed = quantizeFirCoefficientsQ30(hDouble);
    vector<ExperimentRow> rows;
    for (int f = 0; f <= 50; f++){
        rows.push_back(evaluateFrequency(static_cast<double>(f),samplingRate,dur,up,samplingRateOut,taps,beta,guardKaiser,guardFourier,hFixed));
    }
    writeFullCsv(rows, "results.csv");
    return 0;
}