#include <stdio.h>
#include <stdlib.h>

#ifndef NEURAL_NETWORK_ERRORS_LIB
#define NEURAL_NETWORK_ERRORS_LIB

#define err AbsError
#define lenerr LengthError

typedef enum {AbsError, LengthError} ErrorID;

void raiseError(ErrorID id){
	char* errorMessage;
	switch (id){
	case LengthError:
		errorMessage = "LengthError";
		break;
	default:
		errorMessage = "AbsError";
		break;
	}
	printf("%s!\n", errorMessage);
	exit(1);
}

#endif