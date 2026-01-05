//
//  UCUPMultimodal.h
//  UCUP
//
//  Copyright © 2025 UCUP Framework Contributors. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AVFoundation/AVFoundation.h>
#import "UCUP.h"

// Multimodal Input Types
typedef NS_ENUM(NSInteger, UCUPModalityType) {
    UCUPModalityTypeVision,
    UCUPModalityTypeAudio,
    UCUPModalityTypeText,
    UCUPModalityTypeSensor,
    UCUPModalityTypeGeneric
};

// Fusion Strategies
typedef NS_ENUM(NSInteger, UCUPFusionStrategy) {
    UCUPFusionStrategyWeightedAverage,
    UCUPFusionStrategyMaximumConfidence,
    UCUPFusionStrategyConsensus,
    UCUPFusionStrategyConcatenation
};

// Multimodal Input Data
@interface UCUPMultimodalInput : NSObject

@property (nonatomic, assign) UCUPModalityType modality;
@property (nonatomic, copy) NSString *inputId;
@property (nonatomic, strong) NSDictionary *data;
@property (nonatomic, strong) NSDictionary *metadata;
@property (nonatomic, copy) NSString *url;

/**
 * Initialize input
 */
- (instancetype)initWithModality:(UCUPModalityType)modality
                         inputId:(NSString *)inputId
                            data:(NSDictionary *)data;

/**
 * Initialize with URL
 */
- (instancetype)initWithModality:(UCUPModalityType)modality
                         inputId:(NSString *)inputId
                             url:(NSString *)url;

@end

// Vision Processor
@interface UCUPVisionProcessor : NSObject

@property (nonatomic, strong) VNRequestHandler *requestHandler;

/**
 * Initialize vision processor
 */
- (instancetype)init;

/**
 * Process image data
 */
- (NSDictionary *)processImage:(UIImage *)image
                         error:(NSError **)error;

/**
 * Process image from URL
 */
- (NSDictionary *)processImageAtURL:(NSURL *)imageURL
                              error:(NSError **)error;

/**
 * Extract features from image
 */
- (NSArray *)extractFeatures:(UIImage *)image
                     options:(NSDictionary *)options
                       error:(NSError **)error;

@end

// Audio Processor
@interface UCUPAudioProcessor : NSObject

@property (nonatomic, strong) AVAudioEngine *audioEngine;

/**
 * Initialize audio processor
 */
- (instancetype)init;

/**
 * Process audio data
 */
- (NSDictionary *)processAudio:(NSData *)audioData
                     sampleRate:(double)sampleRate
                       channels:(NSInteger)channels
                         error:(NSError **)error;

/**
 * Process audio from URL
 */
- (NSDictionary *)processAudioAtURL:(NSURL *)audioURL
                              error:(NSError **)error;

/**
 * Transcribe speech
 */
- (NSDictionary *)transcribeAudio:(NSData *)audioData
                        language:(NSString *)language
                           error:(NSError **)error;

@end

// Text Processor
@interface UCUPTextProcessor : NSObject

/**
 * Initialize text processor
 */
- (instancetype)init;

/**
 * Analyze text
 */
- (NSDictionary *)analyzeText:(NSString *)text
                        error:(NSError **)error;

/**
 * Extract entities from text
 */
- (NSArray *)extractEntities:(NSString *)text
                     options:(NSDictionary *)options
                       error:(NSError **)error;

/**
 * Classify text sentiment
 */
- (NSDictionary *)classifySentiment:(NSString *)text
                              error:(NSError **)error;

@end

// Sensor Data Processor
@interface UCUPSensorProcessor : NSObject

/**
 * Initialize sensor processor
 */
- (instancetype)init;

/**
 * Process sensor data
 */
- (NSDictionary *)processSensorData:(NSDictionary *)sensorData
                             sensorType:(NSString *)sensorType
                                error:(NSError **)error;

/**
 * Validate sensor data
 */
- (BOOL)validateSensorData:(NSDictionary *)sensorData
                sensorType:(NSString *)sensorType;

/**
 * Calibrate sensor data
 */
- (NSDictionary *)calibrateSensorData:(NSDictionary *)sensorData
                           calibration:(NSDictionary *)calibration;

@end

// Multimodal Fusion Engine
@interface UCUPMultimodalFusionEngine : NSObject

@property (nonatomic, weak) UCUPManager *manager;
@property (nonatomic, strong) UCUPVisionProcessor *visionProcessor;
@property (nonatomic, strong) UCUPAudioProcessor *audioProcessor;
@property (nonatomic, strong) UCUPTextProcessor *textProcessor;
@property (nonatomic, strong) UCUPSensorProcessor *sensorProcessor;

/**
 * Initialize fusion engine
 */
- (instancetype)initWithManager:(UCUPManager *)manager;

/**
 * Fuse multimodal data
 */
- (NSDictionary *)fuseMultimodalData:(NSArray<UCUPMultimodalInput *> *)inputs
                      fusionStrategy:(UCUPFusionStrategy)strategy
                               error:(NSError **)error;

/**
 * Process single modality
 */
- (NSDictionary *)processModality:(UCUPMultimodalInput *)input
                            error:(NSError **)error;

/**
 * Get processing statistics
 */
- (NSDictionary *)getProcessingStats;

/**
 * Reset processing statistics
 */
- (void)resetStats;

@end

// Fusion Result
@interface UCUPFusionResult : NSObject

@property (nonatomic, strong) NSDictionary *fusedAnalysis;
@property (nonatomic, strong) NSDictionary *modalityResults;
@property (nonatomic, assign) UCUPFusionStrategy fusionStrategy;
@property (nonatomic, assign) NSTimeInterval processingTime;
@property (nonatomic, assign) NSInteger inputCount;
@property (nonatomic, strong) NSArray *modalitiesUsed;

/**
 * Initialize result
 */
- (instancetype)initWithFusedAnalysis:(NSDictionary *)fusedAnalysis
                      modalityResults:(NSDictionary *)modalityResults
                       fusionStrategy:(UCUPFusionStrategy)fusionStrategy
                      processingTime:(NSTimeInterval)processingTime
                          inputCount:(NSInteger)inputCount
                      modalitiesUsed:(NSArray *)modalitiesUsed;

@end
