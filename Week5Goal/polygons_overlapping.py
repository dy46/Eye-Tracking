"""
polygons_overlapping.py

This module provides a function for testing two or a set of polygons on 
whether they are overlapping or touching.

Contains:
 * pair_overlapping 
   Finds out if two polygons are overlapping or touching.
 * collection_overlapping
   Finds out which polygons in a set are overlapping or touching.

Author: Fabian Moser http://www.fabianmoser.at

Changelog:
2009-04-30 First release

"""

import numpy as np

class PolygonsTouching( Exception ):
    """ This exception is triggered when two polygons touch at one point.
    
    This is for internal use only and will be caught before returning.
    
    """
    def __init__( self, x=0, y=0 ):
        self.x, self.y = x, y
    def __str__( self ):
        return 'The tested polygons at least touch each other at (%f,%f)'\
               % ( self.x, self.y )
    def shift( self, dx, dy ):
        self.x += dx
        self.y += dy

def pair_overlapping( polygon1, polygon2, digits = None ):
    """ Find out if polygons are overlapping or touching.
    
    The function makes use of the quadrant method to find out if a point is 
    inside a given polygon.
    
    polygon1, polygon2 -- Two arrays of [x,y] pairs where the last and the 
        first pair is the same, because the polygon has to be closed.
    digits -- The number of digits relevant for the decision between 
        separate and touching or touching and overlapping
    
    Returns 0 if the given polygons are neither overlapping nor touching,
    returns 1 if they are not overlapping, but touching and
    returns 2 if they are overlapping
    
    """
    
    def calc_walk_summand( r1, r2, digits = None ):
        """ Calculates the summand along one edge depending on axis crossings.
        
        Follows the edge between two points and checks if one or both axes are
        being crossed. If They are crossed in clockwise sense, it returns +1
        otherwise -1. Going through the origin raises the PolygonsTouching
        exception.
        
        Returns one of -2, -1, 0, +1, +2 or raises PolygonsTouching
        
        """
        x, y = 0, 1 # indices for better readability
        summand = 0 # the return value
        tx, ty = None, None # on division by zero, set parameters to None
        if r1[x] != r2[x]:
            ty = r1[x] / ( r1[x] - r2[x] ) # where it crosses the y axis
        if r1[y] != r2[y]:
            tx = r1[y] / ( r1[y] - r2[y] ) # where it crosses the x axis
        if tx == None: tx = ty
        if ty == None: ty = tx
        rsign = np.sign
        if digits != None:
            rsign = lambda x: np.sign( round( x, digits ) )
        sign_x = rsign( r1[x] + tx * ( r2[x] - r1[x] ) )
        sign_y = rsign( r1[y] + ty * ( r2[y] - r1[y] ) )
        if ( tx >= 0 ) and ( tx < 1 ):
            if ( sign_x == 0 ) and ( sign_y == 0 ):
                raise PolygonsTouching()
            summand += sign_x * np.sign( r2[y] - r1[y] )
        if ( ty >= 0 ) and ( ty < 1 ):
            if ( sign_x == 0 ) and ( sign_y == 0 ):
                raise PolygonsTouching()
            summand += sign_y * np.sign( r1[x] - r2[x] )
        return summand

    def current_and_next( iterable ):
        """ Returns an iterator for each element and its following element.
        
        """
        iterator = iter( iterable )
        item = iterator.next()
        for next in iterator:
            yield ( item, next )
            item = next

    def point_in_polygon( xy, xyarray, digits = None ):
        """ Checks if a point lies inside a polygon using the quadrant method.
        
        This moves the given point to the origin and shifts the polygon
        accordingly. Then for each edge of the polygon, calc_walk_summand is
        called. If the sum of all returned values from these calls is +4 or -4,
        the point lies indeed inside the polygon. Otherwise, if a 
        PolygonsTouching exception has been caught, the point lies on ond of 
        the edges of the polygon.
        
        Returns the number of nodes of the polygon, if the point lies inside,
        otherwise 1 if the point lies on the polygon and if not, 0.
        
        """
        moved = xyarray - xy # move currently checked point to the origin (0,0)
        touching = False # this is used only if no overlap is found
        walk_sum = 0
        for cnxy in current_and_next( moved ):
            try:
                walk_sum += calc_walk_summand( cnxy[0], cnxy[1], digits )
            except PolygonsTouching, (e):
                e.shift( *xy )
                touching = True
        if ( abs( walk_sum ) == 4 ):
            return len( xyarray )
        elif touching:
            return 1
        else:
            return 0

    def polygons_overlapping( p1, p2, digits = None ):
        """ Checks if one of the nodes of p1 lies inside p2.
        
        This repeatedly calls point_in_polygon for each point of polygon p1
        and immediately returns if it is the case, because then the polygons
        are obviously overlapping.
        
        Returns 2 for overlapping polygons, 1 for touching polygons and 0 
        otherwise.
        
        """
        degree_of_contact = 0
        xyarrays = [ p1, p2 ]
        for xy in xyarrays[0]:
            degree_of_contact += point_in_polygon( xy, xyarrays[1], digits )
            if degree_of_contact >= len( xyarrays[1] ):
                return 2
        if degree_of_contact > 0:
            return 1
        else:
            return 0
    
    way1 = polygons_overlapping( polygon1, polygon2, digits )
    way2 = 0
    if way1 < 2: # Only if the polygons are not already found to be overlapping
        way2 = polygons_overlapping( polygon2, polygon1, digits )
    return max( way1, way2 )

def collection_overlapping_serial( polygons, digits = None ):
    """ Similar to the collection_overlapping function, but forces serial 
    processing.
    
    """
    result = []
    pickle_polygons = [p.get_xy() for p in polygons]
    for i in xrange( len( polygons ) ):
        for j in xrange( i+1, len( polygons ) ):
            result.append( ( i, j, \
                pair_overlapping( pickle_polygons[i], pickle_polygons[j], \
                                  digits ) ) )
    return result
    
def __cop_bigger_job( polygons, index, digits = None ):
    """ This is a helper to efficiently distribute workload among processors.
    
    """
    result = []
    for j in xrange( index + 1, len( polygons ) ):
        result.append( ( index, j, \
            pair_overlapping( polygons[index], polygons[j], digits ) ) )
    return result

def collection_overlapping_parallel( polygons, digits = None, \
        ncpus = 'autodetect' ):
    """ Like collection_overlapping, but forces parallel processing.
    
    This function crashes if Parallel Python is not found on the system.
    
    """
    import pp
    ppservers = ()
    job_server = pp.Server( ncpus, ppservers=ppservers )
    pickle_polygons = [p.get_xy() for p in polygons]
    jobs = []
    for i in xrange( len( polygons ) ):
        job = job_server.submit( __cop_bigger_job, \
                                 ( pickle_polygons, i, digits, ), \
                                 ( pair_overlapping, PolygonsTouching, ), \
                                 ( "pylab", ) )
        jobs.append( job )
    result = []
    for job in jobs:
        result += job()
    #job_server.print_stats()
    return result
    
def collection_overlapping( polygons, digits = None ):
    """ Look for pair-wise overlaps in a given list of polygons.
    
    The function makes use of the quadrant method to find out if a point is 
    inside a given polygon. It invokes the pair_overlapping function for each
    combination and produces and array of index pairs of these combinations
    together with the overlap number of that pair. The overlap number is 0 for
    no overlap, 1 for touching and 2 for overlapping polygons.
    
    This function automatically selects between a serial and a parallel 
    implementation of the search depending on whether Parallel Python is 
    installed and can be imported or not.
    
    polygons -- A list of arrays of [x,y] pairs where the last and the first
        pair of each array in the list is the same, because the polygons have 
        to be closed.
    digits -- The number of digits relevant for the decision between 
        separate and touching or touching and overlapping polygons.
    
    Returns a list of 3-tuples
    
    """
    try:
        import pp # try if parallel python is installed
    except ImportError:
        return collection_overlapping_serial( polygons, digits )
    else:
        return collection_overlapping_parallel( polygons, digits )

# if __name__ == '__main__':
#     """ If this module is not imported but itself executed, unit tests are 
#     performed.
    
#     """
#     import unittest, random
    
#     pylab.ioff()

#     class TestCasePolygon( unittest.TestCase ):
#         def setUp( self ):
#             self.candidate = pair_overlapping
#             self.fig = pylab.figure()
#             x = pylab.array( [ 0.0, 0.5, 1.0, 0.5 ] )
#             y = pylab.array( [ 0.5, 0.0, 0.5, 1.0 ] )
#             self.fill1 = pylab.fill( x, y, 'r')
            
#         def tearDown( self ):
#             pylab.close( self.fig )
#             self.fig = None

#     class TestCaseSeparatedPolygons( TestCasePolygon ):
#         def setUp( self ):
#             TestCasePolygon.setUp( self )
#             x = pylab.array( [ 0.5, 1.5, 2.0, 1.0 ] )
#             y = pylab.array( [ 1.5, 0.5, 1.0, 2.0 ] )
#             self.fill2 = pylab.fill( x, y, 'g')
            
#         def testSeparate( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy() )
#             self.assertEqual( degree_of_contact, 0 )

#     class TestCaseTouchingPolygons( TestCasePolygon ):
#         def setUp( self ):
#             TestCasePolygon.setUp( self )
#             x = pylab.array( [ 0.0, 1.0, 1.25, 0.25 ] )
#             y = pylab.array( [ 1.25, 0.75, 1.25, 1.75 ] )
#             self.fill2 = pylab.fill( x, y, 'g')
            
#         def testTouching( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy() )
#             self.assertEqual( degree_of_contact, 1 )
            
#     class TestCaseOverlappingPolygons1( TestCasePolygon ):
#         def setUp( self ):
#             TestCasePolygon.setUp( self )
#             x = pylab.array( [ 0.0, 1.0, 1.0, 0.0 ] )
#             y = pylab.array( [ 0.5, 0.75, 1.5, 1.25 ] )
#             self.fill2 = pylab.fill( x, y, 'g')
            
#         def testOneWayCheck( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy() )
#             self.assertEqual( degree_of_contact, 2 )

#     class TestCaseOverlappingPolygons2( TestCasePolygon ):
#         def setUp( self ):
#             TestCasePolygon.setUp( self )
#             x = pylab.array( [ 0.5, 1.5, 1.5, 1.0 ] )
#             y = pylab.array( [ 0.5, 1.0, 1.5, 1.5 ] )
#             self.fill2 = pylab.fill( x, y, 'g')
            
#         def testTwoWayCheck( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy() )
#             self.assertEqual( degree_of_contact, 2 )

#     class TestCaseAlmostOverlappingPolygons( TestCasePolygon ):
#         def setUp( self ):
#             TestCasePolygon.setUp( self )
#             x = pylab.array( [ 0.5001, 1.0001, 1.5001, 0.5001 ] )
#             y = pylab.array( [ 1.0, 0.5, 1.0, 1.5 ] )
#             self.fill2 = pylab.fill( x, y, 'g')
            
#         def testLowAccuracy( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy(), 3 )
#             self.assertEqual( degree_of_contact, 1 )
            
#         def testHighAccuracy( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy(), 5 )
#             self.assertEqual( degree_of_contact, 0 )
            
#         def testMachineAccuracy( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy() )
#             self.assertEqual( degree_of_contact, 0 )

#     class TestCaseAlmostSeparatePolygons( TestCasePolygon ):
#         def setUp( self ):
#             TestCasePolygon.setUp( self )
#             x = pylab.array( [ 0.4999, 0.9999, 1.4999, 0.4999 ] )
#             y = pylab.array( [ 1.0, 0.5, 1.0, 1.5 ] )
#             self.fill2 = pylab.fill( x, y, 'g')
            
#         def testLowAccuracy( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), \
#                                                 self.fill2[0].get_xy(), 3 )
#             self.assertEqual( degree_of_contact, 1 )
            
#         def testHighAccuracy( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), \
#                                                 self.fill2[0].get_xy(), 5 )
#             self.assertEqual( degree_of_contact, 2 )
            
#         def testMachineAccuracy( self ):
#             degree_of_contact = self.candidate( self.fill1[0].get_xy(), 
#                                                 self.fill2[0].get_xy() )
#             self.assertEqual( degree_of_contact, 2 )

#     class TestCaseLotsOfPolygons( TestCasePolygon ):
#         def setUp( self ):
#             TestCasePolygon.setUp( self )
#             self.fills = [self.fill1]
#             self.n = 100
#             pi = pylab.pi
#             angles = pylab.array( [0.0, 0.5 * pi, 1.0 * pi, 1.5 * pi] )
#             distances = pylab.array( [1.0, 1.2, 1.4, 1.6] )
#             random.seed(0)
#             for i in xrange( self.n - 1 ):
#                 dx = random.uniform( -10.0, 10.0 )
#                 dy = random.uniform( -10.0, 10.0 )
#                 phi = random.uniform( 0.0, 2.0 * pi )
#                 x = pylab.sin( angles + phi ) * distances + dx
#                 y = pylab.cos( angles + phi ) * distances + dy
#                 self.fills.append( pylab.fill( x, y, \
#                         random.choice( 'bgrcmy' ) ) )

#         def testAuto( self ):
#             counter = 0
#             polygons = [f[0] for f in self.fills]
#             found_overlaps = collection_overlapping( polygons )
#             for i, j, degree_of_contact in found_overlaps:
#                 counter += 1
#             self.assertEqual( counter, ( ( self.n - 1 ) * self.n ) / 2 )

#         def testSerial( self ):
#             counter = 0
#             polygons = [f[0] for f in self.fills]
#             found_overlaps = collection_overlapping_serial( polygons )
#             for i, j, degree_of_contact in found_overlaps:
#                 counter += 1
#             self.assertEqual( counter, ( ( self.n - 1 ) * self.n ) / 2 )

#         def testParallel( self ):
#             counter = 0
#             polygons = [f[0] for f in self.fills]
#             found_overlaps = collection_overlapping_parallel( polygons )
#             for i, j, degree_of_contact in found_overlaps:
#                 counter += 1
#             self.assertEqual( counter, ( ( self.n - 1 ) * self.n ) / 2 )
    
#     unittest.main()
