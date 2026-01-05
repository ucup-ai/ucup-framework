//
//  UCUP.h
//  UCUP
//
//  Copyright © 2025 UCUP Framework Contributors. All rights reserved.
//

#import <Foundation/Foundation.h>

//! Project version number for UCUP.
FOUNDATION_EXPORT double UCUPVersionNumber;

//! Project version string for UCUP.
FOUNDATION_EXPORT const unsigned char UCUPVersionString[];

// Core UCUP Classes
@class UCUPManager;
@class UCUPProbabilisticEngine;
@class UCUPCoordinator;
@class UCUPMultimodalProcessor;

// Main framework interface
@interface UCUP : NSObject

/**
 * Initialize the UCUP framework with default configuration
 */
+ (instancetype)sharedInstance;

/**
 * Initialize UCUP with custom configuration
 */
- (instancetype)initWithConfiguration:(NSDictionary *)config;

/**
 * Get the probabilistic reasoning engine
 */
- (UCUPProbabilisticEngine *)probabilisticEngine;

/**
 * Get the coordination system
 */
- (UCUPCoordinator *)coordinator;

/**
 * Get the multimodal processor
 */
- (UCUPMultimodalProcessor *)multimodalProcessor;

/**
 * Execute probabilistic reasoning with Grand Central Dispatch optimization
 */
- (void)executeProbabilisticTask:(NSDictionary *)parameters
                 completionHandler:(void (^)(NSDictionary *result, NSError *error))completionHandler;

/**
 * Perform multimodal analysis using Core ML acceleration
 */
- (void)analyzeMultimodalData:(NSArray *)data
              completionHandler:(void (^)(NSDictionary *analysis, NSError *error))completionHandler;

@end

// Core ML Integration
@interface UCUPProbabilisticEngine : NSObject

- (instancetype)initWithCoreMLModel:(MLModel *)model;
- (NSDictionary *)predict:(NSDictionary *)input error:(NSError **)error;

@end

// Grand Central Dispatch Coordination
@interface UCUPCoordinator : NSObject

- (void)coordinateTasks:(NSArray *)tasks
       withConcurrency:(NSUInteger)concurrency
    completionHandler:(void (^)(NSArray *results, NSError *error))completionHandler;

@end

// Vision and AVFoundation Integration
@interface UCUPMultimodalProcessor : NSObject

- (void)processImage:(UIImage *)image
    completionHandler:(void (^)(NSDictionary *features, NSError *error))completionHandler;

- (void)processAudio:(NSURL *)audioURL
    completionHandler:(void (^)(NSDictionary *transcription, NSError *error))completionHandler;

@end
