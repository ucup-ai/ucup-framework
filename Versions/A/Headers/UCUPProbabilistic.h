//
//  UCUPProbabilistic.h
//  UCUP
//
//  Copyright © 2025 UCUP Framework Contributors. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "UCUP.h"

// Probabilistic Reasoning Classes
@interface UCUPProbabilisticEngine : NSObject

/**
 * Initialize probabilistic engine with optional Core ML model
 */
- (instancetype)initWithModelPath:(NSString *)modelPath;

/**
 * Make probabilistic prediction
 */
- (NSDictionary *)predict:(NSDictionary *)input error:(NSError **)error;

/**
 * Evaluate uncertainty in predictions
 */
- (NSDictionary *)evaluateUncertainty:(NSArray *)predictions;

@end

@interface UCUPProbabilisticAgent : NSObject

/**
 * Initialize agent with manager
 */
- (instancetype)initWithManager:(UCUPManager *)manager;

/**
 * Make decision based on context and options
 */
- (NSDictionary *)makeDecision:(NSDictionary *)context
                       options:(NSArray *)options;

/**
 * Learn from decision outcomes
 */
- (void)learnFromOutcome:(NSString *)decisionId
               actualOutcome:(NSNumber *)outcome
                   feedback:(NSDictionary *)feedback;

/**
 * Get decision history
 */
- (NSArray *)getDecisionHistory;

@end

// Probabilistic Result Container
@interface UCUPProbabilisticResult : NSObject

@property (nonatomic, readonly) NSNumber *probability;
@property (nonatomic, readonly) NSArray *confidenceInterval;
@property (nonatomic, readonly) NSArray *alternativePaths;
@property (nonatomic, readonly) NSNumber *uncertaintyMeasure;
@property (nonatomic, readonly) NSString *processingMethod;
@property (nonatomic, readonly) NSDate *timestamp;

/**
 * Initialize result
 */
- (instancetype)initWithProbability:(NSNumber *)probability
                  confidenceInterval:(NSArray *)confidenceInterval
                    alternativePaths:(NSArray *)alternativePaths
                  uncertaintyMeasure:(NSNumber *)uncertaintyMeasure
                   processingMethod:(NSString *)processingMethod;

/**
 * Check if result meets confidence threshold
 */
- (BOOL)isConfident:(NSNumber *)threshold;

/**
 * Get best alternative path
 */
- (NSDictionary *)getBestAlternative;

@end
