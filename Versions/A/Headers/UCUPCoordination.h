//
//  UCUPCoordination.h
//  UCUP
//
//  Copyright © 2025 UCUP Framework Contributors. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "UCUP.h"

// Coordination Strategy Types
typedef NS_ENUM(NSInteger, UCUPCoordinationStrategy) {
    UCUPCoordinationStrategyHierarchical,
    UCUPCoordinationStrategySwarm,
    UCUPCoordinationStrategyAdaptive
};

// Task Definition
@interface UCUPTask : NSObject

@property (nonatomic, copy) NSString *taskId;
@property (nonatomic, copy) NSString *type;
@property (nonatomic, strong) NSDictionary *parameters;
@property (nonatomic, strong) NSDictionary *data;
@property (nonatomic, assign) NSInteger priority;
@property (nonatomic, copy) NSString *parentId;
@property (nonatomic, strong) NSArray<NSString *> *dependencies;

/**
 * Initialize task
 */
- (instancetype)initWithId:(NSString *)taskId
                      type:(NSString *)type
                parameters:(NSDictionary *)parameters;

@end

// Base Coordinator Protocol
@protocol UCUPCoordinator <NSObject>

@required
- (NSDictionary *)coordinateTasks:(NSArray<UCUPTask *> *)tasks
                            error:(NSError **)error;

@optional
- (NSDictionary *)getCoordinatorStatus;
- (void)cancelExecution;

@end

// Hierarchical Coordinator
@interface UCUPHierarchicalCoordinator : NSObject <UCUPCoordinator>

@property (nonatomic, weak) UCUPManager *manager;

/**
 * Initialize hierarchical coordinator
 */
- (instancetype)initWithManager:(UCUPManager *)manager;

/**
 * Coordinate tasks hierarchically
 */
- (NSDictionary *)coordinateTasks:(NSArray<UCUPTask *> *)tasks
                            error:(NSError **)error;

/**
 * Get task execution status
 */
- (NSDictionary *)getTaskStatus:(NSString *)taskId;

/**
 * Get hierarchy status
 */
- (NSDictionary *)getHierarchyStatus;

@end

// Swarm Coordinator
@interface UCUPSwarmCoordinator : NSObject <UCUPCoordinator>

@property (nonatomic, weak) UCUPManager *manager;
@property (nonatomic, assign) NSInteger numAgents;

/**
 * Initialize swarm coordinator
 */
- (instancetype)initWithManager:(UCUPManager *)manager
                      numAgents:(NSInteger)numAgents;

/**
 * Coordinate tasks using swarm intelligence
 */
- (NSDictionary *)coordinateTasks:(NSArray<UCUPTask *> *)tasks
                            error:(NSError **)error;

/**
 * Get swarm status
 */
- (NSDictionary *)getSwarmStatus;

/**
 * Update swarm parameters
 */
- (void)updateParameters:(NSDictionary *)parameters;

@end

// Adaptive Coordinator
@interface UCUPAdaptiveCoordinator : NSObject <UCUPCoordinator>

@property (nonatomic, weak) UCUPManager *manager;
@property (nonatomic, strong) UCUPHierarchicalCoordinator *hierarchicalCoordinator;
@property (nonatomic, strong) UCUPSwarmCoordinator *swarmCoordinator;

/**
 * Initialize adaptive coordinator
 */
- (instancetype)initWithManager:(UCUPManager *)manager;

/**
 * Coordinate tasks with adaptive strategy selection
 */
- (NSDictionary *)coordinateTasks:(NSArray<UCUPTask *> *)tasks
                            error:(NSError **)error;

/**
 * Get adaptation status and metrics
 */
- (NSDictionary *)getAdaptationStatus;

/**
 * Force specific strategy
 */
- (void)forceStrategy:(UCUPCoordinationStrategy)strategy;

/**
 * Reset learning state
 */
- (void)resetLearning;

@end

// Coordination Result
@interface UCUPCoordinationResult : NSObject

@property (nonatomic, assign) BOOL success;
@property (nonatomic, strong) NSArray *results;
@property (nonatomic, assign) NSTimeInterval executionTime;
@property (nonatomic, strong) NSDictionary *metadata;
@property (nonatomic, strong) NSError *error;

/**
 * Initialize result
 */
- (instancetype)initWithSuccess:(BOOL)success
                         results:(NSArray *)results
                   executionTime:(NSTimeInterval)executionTime
                        metadata:(NSDictionary *)metadata
                          error:(NSError *)error;

@end
