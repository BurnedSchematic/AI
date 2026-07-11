/*
=========================================================================================
DIGITAL SIGNATURE: 
Signed: Sigge Robert Dahl
Handle: BurnedSchematic
Date Signed: 7/7/2026
Date Finished: 7/8/2026
Witness:
Credits:
Main Framework (Classes and functions): Sigge Dahl
Debugging and Saving framework: OpenAI ChatGPT
Debugging and main AI loop: Microsoft Copilot
=========================================================================================
*/
#include <functional>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <filesystem>
#include <portaudio.h>
#include <fstream>
#include <unordered_map>

using namespace std;
using std::clamp;

int muscleCount = 128 + 2;//pixels in out + frequency and volume of audio out

vector<neuron> brain;
vector<InputNeuron> sensoryNerves;
vector<neuron> muscles;

#define SAMPLE_RATE 48000

//define callback portaudio for input/output (audio)

typedef struct
{
    int frameIndex;

    float latestInput;

    float frequency;
    float phase;
    float amplitude;
}
paTestData;

static int audioCallback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo *timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData) 
{
    const float *in = (const float *)inputBuffer;
    float *out = (float *)outputBuffer;
    paTestData *data = (paTestData *)userData;

    for(unsigned long i = 0; i < framesPerBuffer; i++)
    {
        float micSample = in ? in[i] : 0.0f;

        data->latestInput = micSample;

        float speakerSample =
            data->amplitude * sinf(2.0f * M_PI * data->phase);

        out[i] = speakerSample;

        data->phase += data->frequency / SAMPLE_RATE;

        if(data->phase >= 1.0f) data->phase -= 1.0f;
    }
    return paContinue;
}

//define a function for playing sound
void playFrequency(paTestData* data, float frequency, float volume)
{
    data->frequency = frequency;
    data->amplitude = volume;
}


//create framebuffer for initialization as well as for HDMI drawing

struct framebuffer
{
    fb_var_screeninfo vinfo;
    fb_fix_screeninfo finfo;
    char* fbp;
    int fbfd;
    long screensize;
    bool init()
    {
        fbfd = open("/dev/fb0", O_RDWR);
        ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo);
        ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo);
        screensize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;
        fbp = (char *)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fbfd, 0);
        return true;
    }

    void drawPixel(int x, int y, vector<int> rgba)
    {
        // For 32-bit (RGBA) screen
        long location = (x + vinfo.xoffset) * (vinfo.bits_per_pixel / 8) +
                (y + vinfo.yoffset) * finfo.line_length;
        *(fbp + location) = rgba[2];     // Blue
        *(fbp + location + 1) = rgba[1];   // Green
        *(fbp + location + 2) = rgba[0];   // Red
        *(fbp + location + 3) = rgba[3];   // Alpha

    }
};

vector<int> getFourUnbiasedNumbers(int input) {
    // 1. Generate a deterministic, well-distributed 64-bit hash of the input
    hash<int> hasher;
    size_t hashValue = hasher(input);
    
    // 2. Extract 4 independent, non-overlapping bytes (8 bits each)
    // Masking with 0xFF ensures the output is always between 0 and 255
    int num1 = (hashValue >> 0)  & 0xFF;
    int num2 = (hashValue >> 8)  & 0xFF;
    int num3 = (hashValue >> 16) & 0xFF;
    int num4 = (hashValue >> 24) & 0xFF;
    
    return {num1, num2, num3, num4};
}

//define average neuron
//Can connect, check if it is activated by surrounding neurons, and reinitialize its values to learn

struct neuron
{
    int id;

    bool activated = false;
    vector<neuron*> inputConnections;
    vector<int> inputStrengths;
    vector<int> ageSinceLastUse;
    vector<neuron*> outputConnections;
    int threshold = 55;
    int outputValue = 55;
    bool checkActive()
    {
        int tally = 0;

        for(int i = 0; i < inputConnections.size(); i++)
        {
            if(inputConnections[i]->activated == false)
            {
                ageSinceLastUse[i] += 1;
                continue;
            }
            tally += inputConnections[i]->outputValue*inputStrengths[i];
        }
        if(tally > threshold)
        {
            for(int i = 0; i < inputConnections.size(); i++)
            {
                if(ageSinceLastUse[i] < *max_element(ageSinceLastUse.begin(), ageSinceLastUse.end())-1)
                {
                    inputStrengths[i] += 1;
                } else {
                    inputStrengths[i] -= 1;
                }
                inputStrengths[i] = clamp(inputStrengths[i], -100, 100);
            }
            activated = true;
            return true;
        } else {
            activated = false;
            return false;
        }
    }
    void connect(neuron* newConnection)
    {
        outputConnections.push_back(newConnection);
        newConnection->inputStrengths.push_back(0);
        newConnection->ageSinceLastUse.push_back(0);
        newConnection->inputConnections.push_back(this);
    }
    void hebbianUpdate(float learningRate = 0.01f)
    {
        // For each input connection
        for (int i = 0; i < inputConnections.size(); i++)
        {
            neuron* pre = inputConnections[i];   // presynaptic neuron
            neuron* post = this;                 // postsynaptic neuron

            // Hebbian rule: Δw = η * pre.activation * post.activation
            if (pre->activated && post->activated)
            {
                inputStrengths[i] += learningRate * 100; // scale to your -100..100 range
            }
            else
            {
                // Optional: anti-Hebbian decay
                inputStrengths[i] -= learningRate * 10;
            }

            // Clamp to your allowed range
            inputStrengths[i] = clamp(inputStrengths[i], -100, 100);
        }
    }
    void reInit()
    {
        threshold = clamp(threshold, 1, 1000);
        if(inputStrengths.empty()) {
            return;
        }
        if(accumulate(inputStrengths.begin(), inputStrengths.end(), 0.0)/inputStrengths.size() < 1)
        {
            threshold -= 1;
        } else if(accumulate(inputStrengths.begin(), inputStrengths.end(), 0.0)/inputStrengths.size() >= 55)
        {
            threshold += 1;
        }
        vector<int> otherInputStrengths;

        for(int i = 0; i < outputConnections.size(); i++)
        {
            otherInputStrengths.insert(otherInputStrengths.end(), outputConnections[i]->inputStrengths.begin(), outputConnections[i]->inputStrengths.end());
        }
        if(otherInputStrengths.empty()) return;
        if(accumulate(otherInputStrengths.begin(), otherInputStrengths.end(), 0.0)/otherInputStrengths.size() < 1)
        {
            outputValue += 1;
        } else if(accumulate(otherInputStrengths.begin(), otherInputStrengths.end(), 0.0)/otherInputStrengths.size() >= 55)
        {
            outputValue -= 1;
        }
    }
};

//define input neuron to get inputs from raw integers instead of other neurons

struct InputNeuron
{
    int id;

    vector<neuron*> outputConnections;
    
    bool activated = false;
    int threshold = 55;
    int outputValue = 55;
    bool checkActive(vector<int> inputs)
    {
        if(accumulate(inputs.begin(), inputs.end(), 0) > threshold)
        {
            outputValue = accumulate(inputs.begin(), inputs.end(), 0);
            activated = true;
            return true;
        }
        activated = false;
        return false;
    }
    void connect(neuron* newConnection)
    {
        outputConnections.push_back(newConnection);
    }
    void reInit()
    {
        threshold = clamp(threshold, 1, 1000);
        vector<int> otherInputStrengths;
        for(int i = 0; i < outputConnections.size(); i++)
        {
            otherInputStrengths.insert(otherInputStrengths.end(), outputConnections[i]->inputStrengths.begin(), outputConnections[i]->inputStrengths.end());
        }
        if(otherInputStrengths.empty()) return;
        if(accumulate(otherInputStrengths.begin(), otherInputStrengths.end(), 0.0)/otherInputStrengths.size() < 1)
        {
            threshold -= 1;
        } else if(accumulate(otherInputStrengths.begin(), otherInputStrengths.end(), 0.0)/otherInputStrengths.size() >= 55)
        {
            threshold += 1;
        }
    }
};

//Network saving for if bot gets unplugged

void saveNetwork(vector<neuron>& neurons)
{
    ofstream file("brain.dat");

    file << neurons.size() << "\n";

    for(auto& n : neurons)
    {
        file << n.id << " ";
        file << n.threshold << " ";
        file << n.outputValue << " ";

        file << n.inputConnections.size() << " ";
        file << n.outputConnections.size() << " ";

        for(int i = 0; i < n.inputConnections.size(); i++)
        {
            file << n.inputConnections[i]->id << " ";
            file << n.inputStrengths[i] << " ";
        }

        for(int i = 0; i < n.outputConnections.size(); i++)
        {
            file << n.outputConnections[i]->id << " ";
        }

        file << "\n";
    }
}

//define function to load the network if returning to power

void loadNetwork(vector<neuron>& neurons)
{
    if (!filesystem::exists("brain.dat"))
    {
        return;
    }
    ifstream file("brain.dat");

    int neuronCount;
    file >> neuronCount;

    neurons.resize(neuronCount);

    unordered_map<int, neuron*> idMap;

    // Create neurons
    for(int i = 0; i < neuronCount; i++)
    {
        neurons[i].id = i;
        idMap[i] = &neurons[i];
    }

    // Load data
    for(int i = 0; i < neuronCount; i++)
    {
        neuron& n = neurons[i];

        int connectionCount;
        int outputConnectionCount;

        file >> n.id;
        file >> n.threshold;
        file >> n.outputValue;
        file >> connectionCount;
        file >> outputConnectionCount;

        for(int j = 0; j < connectionCount; j++)
        {
            int targetID;
            int strength;

            file >> targetID;
            file >> strength;

            n.inputConnections.push_back(idMap[targetID]);
            n.inputStrengths.push_back(strength);
            n.ageSinceLastUse.push_back(0);
        }

        for(int j = 0; j < outputConnectionCount; j++)
        {
            int targetID;

            file >> targetID;

            n.outputConnections.push_back(idMap[targetID]);
        }
    }
    for(int i = 0; i < muscleCount; i++)
    {
        muscles.push_back(brain[i]);
    }
}

void initializeNetwork(int neuronCount, int inputCount)
{
    brain.clear();
    sensoryNerves.clear();
    muscles.clear();

    // Create normal neurons
    for (int i = 0; i < neuronCount; i++)
    {
        neuron n;
        n.id = i;
        brain.push_back(n);
    }

    // Create input neurons
    for (int i = 0; i < inputCount; i++)
    {
        InputNeuron in;
        in.id = i;
        sensoryNerves.push_back(in);
    }

    // Randomly connect input neurons → brain neurons
    for (auto& in : sensoryNerves)
    {
        for (int j = 0; j < 5; j++) // each input neuron connects to 5 brain neurons
        {
            int target = rand() % brain.size();
            in.connect(&brain[target]);
        }
    }

    // Randomly connect brain neurons to each other
    for (auto& n : brain)
    {
        for (int j = 0; j < 3; j++) // each neuron connects to 3 others
        {
            int target = rand() % brain.size();
            if (target != n.id)
                n.connect(&brain[target]);
        }
    }
    for(int i = 0; i < muscleCount; i++)
    {
        muscles.push_back(brain[i]);
    }
}


int main()
{
    framebuffer fb;
    //Initialize network
    loadNetwork(brain);

    if (brain.empty())
    {
        initializeNetwork(500, 50); // 500 neurons, 50 input neurons
    }

    static paTestData data;
    data.frequency = 0.0f;
    data.phase = 0.0f;
    data.amplitude = 0.0f;

    PaError err;

    err = Pa_Initialize();

    PaStreamParameters inputParameters;
    PaStreamParameters outputParameters;
    PaStream *stream;

    // Configure input
    inputParameters.device = Pa_GetDefaultInputDevice();
    inputParameters.channelCount = 1; 
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    // Configure output
    outputParameters.device = Pa_GetDefaultOutputDevice();
    outputParameters.channelCount = 1;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;

    

    fb.init();

    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened()) 
    {
        cerr << "Could not open camera." << endl;
        return 0;
    }

    cv::Mat frame;

    err = Pa_OpenStream(
        &stream,
        &inputParameters,
        &outputParameters,
        SAMPLE_RATE,
        paFramesPerBufferUnspecified,
        paNoFlag,
        audioCallback, //callback function
        &data
        );
    err = Pa_StartStream( stream );

    //Run AI
    while (true)
    {
        cap >> frame;
        vector<int> img;

        if(frame.empty()) {break;}

        float latestAudio =
        data.latestInput;
        for (int r = 0; r < frame.rows; r++)
        {
            uchar* row_ptr = frame.ptr<uchar>(r);
            for (int c = 0; c < frame.cols; c++)
            {
                int b = row_ptr[c*3 + 0];
                int g = row_ptr[c*3 + 1];
                int r_ = row_ptr[c*3 + 2];
                img.push_back(r_+(r_ + g + 1)+(r_ + g + b + 2));//Insert a new number specific to those pixel values into the image list       
            }
        }

        //audio input is latestAudio
        //visual input is img[pixel]

        //AI code

        for (int i = 0; i < sensoryNerves.size(); i++)
        {
            vector<int> inputs;

            // Visual input
            if (i < img.size())
                inputs.push_back(img[i]);

            // Audio input (same for all input neurons)
            inputs.push_back((int)(latestAudio * 1000));

            sensoryNerves[i].checkActive(inputs);
        }
        for (auto& n : brain)
        {
            n.checkActive();      // compute activation
        }

        for (auto& n : brain)
        {
            n.hebbianUpdate();    // update weights based on activation
        }

        for (auto& n : brain)
        {
            n.reInit();           // homeostasis logic
        }
        
        //output visual and audio from muscle neurons
        int x = 0;
        int y = 0;

        for(int i = 0; i < muscleCount-2; i++)
        {
            x++;
            if (x >= 16) {
                x = 0;
                y++;
            }
            if (y >= 8) y = 0;
            if(i < 128)
            {
                if(muscles[i].activated)
                {
                    fb.drawPixel(x,y,getFourUnbiasedNumbers(muscles[i].outputValue));
                } else {
                    fb.drawPixel(x,y,{0,0,0,0});
                }
            } else {
                if(muscles[i].activated && muscles[i+1].activated)
                {
                    playFrequency(data,muscles[i].outputValue,muscles[i+1].outputValue);
                }
            }
        }
        //Stop AI code/repeat
    }
    err = Pa_CloseStream(stream);
    munmap(fb.fbp, fb.screensize);
    close(fb.fbfd);
    err = Pa_Terminate();
    return 0;
}